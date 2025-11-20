import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.tasks import repeat_every
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from utils.isbn_validations import EnhancedPOSProcessor,HCColUpdates,DatabaseTableManager,PriceLogicProcessor,DateOperationsProcessor,Miscellaneous
import config
from logger import logger
from services.azure_synapse import get_mi_connection_string
from utils.isbn_validations import source_configurations
import traceback
from routers import (
    ai_suggestion,
    ai_transformation,
    bigquery_routes,
    downlaoder,
    is_live,
    mongo_db,
    misc,
    annotations,
    scheduler,
    chat_assistant,
    alert_email,
    ai_data_validation,
    schema_management,
    user,
    data_feed_flow,
    application_configurations,
)
from datetime import datetime, time, timedelta
import pytz
from Enums.platform_type import PlatformType
from services.automation.scheduler import create_channels_for_all_folders
from db.mongodb import MongoDBConnector
from logger import logger
import time as time_module
from services.redis_service import redis_client
CACHE_DIR = "cache"
class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int = 300):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
    
    async def dispatch(self, request: Request, call_next):
        try:
            # Set timeout for the request
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Request timeout after {self.timeout_seconds} seconds for {request.url.path}")
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Request timeout",
                    "detail": f"Request exceeded {self.timeout_seconds} seconds"
                }
            )
        except Exception as e:
            logger.error(f"Error in timeout middleware: {str(e)}")
            raise
async def daily_refresh_scheduler(db_manager: "DatabaseTableManager"):
    ET_TIMEZONE = pytz.timezone('America/New_York')
    TARGET_TIME = time(2, 0)

    while True:
        try:
            now_et = datetime.now(ET_TIMEZONE)
            
            if now_et.time() >= TARGET_TIME:
                next_run_date = now_et.date() + timedelta(days=1)
            else:
                next_run_date = now_et.date()

            next_run_datetime = ET_TIMEZONE.localize(datetime.combine(next_run_date, TARGET_TIME))
            
            wait_seconds = (next_run_datetime - now_et).total_seconds()
            
            logger.info(f"Next cache refresh scheduled for {next_run_datetime}. Waiting for {wait_seconds:.0f} seconds.")
            await asyncio.sleep(wait_seconds)

            logger.info("It's 12 AM ET. Kicking off scheduled refresh of lookup tables...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await asyncio.to_thread(db_manager.refresh_lookup_tables, source_configurations)
                    redis_conn = redis_client.get_connection()
                    
                    await redis_conn.set("cache_last_refreshed_timestamp", time_module.time())
                    logger.info("Updated cache refresh timestamp in Redis.")
                    logger.info("Daily cache refresh completed successfully.")
                    break
                except Exception as e:
                    logger.error(f"Refresh Attempt {attempt + 1} failed: {e}", exc_info=True)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"An error occurred in the daily refresh scheduler: {e}")
            logger.error(f"Traceback:{traceback.print_exc()}")
            await asyncio.sleep(300)
async def load_data_in_background(app: FastAPI):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            connection_str = get_mi_connection_string()
            db_manager = DatabaseTableManager(connection_str, logger=logger)
            # Make sure 'source_configs' is defined or imported here
            await asyncio.to_thread(db_manager.warm_up_cache, source_configurations)
            app.state.db_manager = db_manager
            logger.info("db_manager created. Starting daily refresh scheduler...")
            refresh_task = asyncio.create_task(
                daily_refresh_scheduler(app.state.db_manager)
            )
        # Store the task handle so it can be cancelled on shutdown
            app.state.refresh_task = refresh_task
            logger.info("Initializing application processors...")
            
            app.state.pos_processor = await asyncio.to_thread(
                EnhancedPOSProcessor, connection_string=connection_str, db_manager=db_manager
            )
            app.state.col_updates_processor = await asyncio.to_thread(
                HCColUpdates, connection_string=connection_str, db_manager=db_manager
            )
            # app.state.price_logic_processor = await asyncio.to_thread(
            #     PriceLogicProcessor, connection_string=connection_str, db_manager=db_manager
            # )
            app.state.date_logic_processor = await asyncio.to_thread(
                DateOperationsProcessor, connection_string=connection_str, db_manager=db_manager
            )
            # app.state.misc_logic_processor = await asyncio.to_thread(
            #     Miscellaneous, connection_string=connection_str, db_manager=db_manager
            # )
            
            app.state.initialization_event.set()
            logger.info("Background initialization complete. Processors are ready.")
            break
        except asyncio.TimeoutError:
            logger.error(f"Attempt {attempt + 1} timed out")
            if attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
            import shutil
            import os
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                logger.info("Cache directory cleared")
            if attempt < max_retries - 1:
                await asyncio.sleep(5 * (attempt + 1))
            else:
                app.state.initialization_error = e
                logger.error("❌ All initialization attempts failed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the FastAPI application
    """
    try:
        MongoDBConnector.connect(
            connection_string=config.MONGO_URI,
            db_name=config.DATABASE
        )
        if config.FLOW=='hc-flow':
            app.state.initialization_event = asyncio.Event()
            app.state.initialization_error = None

            # Start the slow data loading process in the background
            asyncio.create_task(load_data_in_background(app))
            
        await redis_client.connect()
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield  # Application runs here

    # Shutdown
    try:
        # Close MongoDB connection
        MongoDBConnector.close()
        logger.info("MongoDB connection closed successfully")
        # if config.PLATFORM==PlatformType.AZURE.value:
        await redis_client.disconnect()
    except Exception as e:
        logger.info(f"Error during shutdown: {e}")

def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    application = FastAPI(title="DOC-AI Studio backend APIs", version="1.0.0",lifespan=lifespan)
    application.add_middleware(TimeoutMiddleware, timeout_seconds=600)
    application.add_middleware(
        CORSMiddleware,  # noqa
        allow_origins=["https://doc-ai-studio-dev.techo.camp", 
                       "https://doc-ai-studio.techo.camp",
                       "http://localhost:8000",
                       "https://local.docaistudio.techo.camp:5173",
                       "https://local.cw.techo.camp:3000",
                       "https://dev-creative-workspace.techo.camp",
                       "https://demo.creativeworkspace.ai",
                       "https://datamodaidev.techo.camp",
                       "https://dev.appmod.ai",
                       "https://dev-cw-ai-studio-frontend.techo.camp",
                       "https://appmod.ai",
                       "https://dev-hc-cw.data-studio.ai",
                       "https://dev-hc.data-studio.ai"],
        allow_credentials=True,
        allow_methods=["*"],  
        allow_headers=["*"],  
    )

    application.include_router(downlaoder.router, prefix="/api",
                               tags=["download-sources"])
    application.include_router(is_live.router, prefix="/api", tags=["live"])
    application.include_router(mongo_db.router, prefix="/db", tags=["Database"])
    application.include_router(misc.router, prefix="/api", tags=["Misc"])
    application.include_router(annotations.router, prefix="/api",
                               tags=["Annotation"])
    application.include_router(ai_suggestion.router, prefix="/api",
                               tags=["AI Labels Suggestion"])
    application.include_router(bigquery_routes.router, prefix="/api",
                               tags=["Google Cloud Services"])
    application.include_router(scheduler.router, prefix="/api",
                               tags=["Automation Pipeline"])
    application.include_router(chat_assistant.router, prefix="/chat",
                               tags=["Chat Assistant"])
    application.include_router(alert_email.router, prefix="/mail", tags=["Alert System"])

    application.include_router(ai_transformation.router, prefix="/api", tags=["AI Cleaning"]) 

    application.include_router(ai_data_validation.router, prefix="/api", tags=["AI Validation"])
       
    application.include_router(user.router, prefix="/users", tags=["Users"])

    application.include_router(data_feed_flow.router, prefix="/api", tags=["Data feed flow"])
    application.include_router(application_configurations.router, prefix="/config", tags=["Application Configurations"])
    application.include_router(schema_management.router, prefix="/api", tags=["Target Schema"])
    return application


app = create_application()

async def run_in_background(func: Any):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, func)

         
if __name__ == "__main__":
    import uvicorn
    logger.info("********** Application started **********")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

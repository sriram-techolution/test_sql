import os
import urllib.parse
from celery import Celery
from dotenv import load_dotenv
import sys
from celery.signals import worker_process_init
from celery.signals import worker_ready
import config
from redis import Redis
from types import SimpleNamespace
import time
import logging
import time
import asyncio
from utils.isbn_validations import source_configurations
from utils import worker_state
from logger import logger
from services.azure_synapse import get_mi_connection_string
from utils.isbn_validations import EnhancedPOSProcessor,HCColUpdates,DatabaseTableManager,PriceLogicProcessor,DateOperationsProcessor,Miscellaneous
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


host = config.REDIS_HOST
port = config.REDIS_PORT
db = config.REDIS_DB
password = config.REDIS_PASSWORD
redis_password_part = f":{urllib.parse.quote_plus(password)}" if password else ""
protocol = "redis://"
credentials = f":{password}@" if password else ""
redis_url = f"{protocol}{credentials}{host}:{port}/{db}"

celery_app = Celery("doc_ai_tasks")

celery_app.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    include=[
        "services.automation.scheduler",
        "services.automation.folder_cleanup",
        "services.automation.data_lake_scheduler",
        "routers.scheduler"
    ],
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1,
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)

celery_app.conf.beat_schedule = {
    'run-db-scheduler-every-minute': {
        'task': 'services.automation.scheduler.database_ticker',
        'schedule': 60.0,  # Run every 60 seconds
    },
}



# @worker_process_init.connect
@worker_ready.connect
def initialize_processors(sender=None, **kwargs):
    """
    This function runs when the worker is ready (works with both process and thread pools).
    """
    max_retries = 3
    
    for attempt in range(max_retries):
      try:
        logger.info("Initializing data processors for Celery worker...")
        if not hasattr(celery_app, 'state'):
            celery_app.state = SimpleNamespace()
        try:
            protocol = "redis://"
            credentials = f":{config.REDIS_PASSWORD}@" if config.REDIS_PASSWORD else ""
            redis_url = f"{protocol}{credentials}{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}"
            
            # Create a synchronous Redis client for the Celery worker
            redis_client = Redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            celery_app.state.redis_client = redis_client
            logger.info("Synchronous Redis client connected for Celery worker.")
        except Exception as e:
            logger.error(f"Failed to connect Celery worker to Redis: {e}")
            celery_app.state.redis_client = None
        conn_str = get_mi_connection_string()
        db_manager = DatabaseTableManager(connection_string=conn_str, logger=logger)
        db_manager.warm_up_cache(source_configs=source_configurations)
        celery_app.state.db_manager = db_manager

        celery_app.state.cache_timestamp = time.time()
        celery_app.state.pos_processor_instance = EnhancedPOSProcessor(connection_string=conn_str,db_manager=db_manager)
        # celery_app.state.price_logic_instance = PriceLogicProcessor(connection_string=conn_str,db_manager=db_manager)
        celery_app.state.col_updates_instance = HCColUpdates(connection_string=conn_str,db_manager=db_manager)
        celery_app.state.end_dates_instance = DateOperationsProcessor(connection_string=conn_str,db_manager=db_manager)
        # celery_app.state.misc_updates_instance = Miscellaneous(connection_string=conn_str,db_manager=db_manager)
        print(f"{celery_app.state.pos_processor_instance=}")
        logger.info("Celery worker processors are ready.")
      except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                backoff_time = 5 * (attempt + 1)
                logger.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)



__all__ = ['celery_app']
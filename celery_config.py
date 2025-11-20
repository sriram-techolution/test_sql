from celery import Celery
from celery.schedules import crontab
import sys, os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

celery_app = Celery(
    "doc_ai_tasks",
    broker=f"sqla+sqlite:///{os.path.join(os.getcwd(), 'celery_broker.sqlite')}"
)

celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.timezone = "UTC"
celery_app.conf.enable_utc = True
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.worker_max_tasks_per_child = 1  
celery_app.conf.task_acks_late = True
celery_app.conf.broker_connection_retry_on_startup = True  

celery_app.conf.imports = (
    "services.automation.scheduler",
    "services.automation.folder_cleanup",
    "services.automation.data_lake_scheduler"
)

celery_app.conf.beat_schedule = {
    "clean-old-folders-every-18-hours": {
        "task": "services.automation.folder_cleanup.clean_old_folders",  
         "schedule": crontab(minute="0", hour="*/16"), 
    },
     "recreate-channels-every-20-hours": {
        "task": "services.automation.scheduler.create_channels_for_all_folders",
        "schedule": crontab(minute="0", hour="*/20"),
    },
}
from celery import Celery

from app.core.config import get_settings


settings = get_settings()

celery_app = Celery(
    "zxl_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.document_tasks"],
)

celery_app.conf.update(
    task_serializer=settings.celery_task_serializer,
    result_serializer=settings.celery_result_serializer,
    accept_content=[settings.celery_accept_content],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
)


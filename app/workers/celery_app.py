"""
Celery application instance and configuration.

Configured with Redis as both broker and result backend.
Task serialization uses JSON for cross-language compatibility.
"""

from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "reconface",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,

    # Result expiry
    result_expires=3600,  # 1 hour

    # Worker settings
    worker_prefetch_multiplier=1,  # Fair scheduling for long-running tasks
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to free memory
    worker_concurrency=4,

    # Task routing
    task_default_queue="reconface",
    task_routes={
        "app.workers.tasks.ingest_image_task": {"queue": "ingest"},
        "app.workers.tasks.batch_ingest_task": {"queue": "ingest"},
    },

    # Retry policy
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Auto-discover tasks in the workers package
celery_app.autodiscover_tasks(["app.workers"])

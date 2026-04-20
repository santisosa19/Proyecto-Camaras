"""
Instancia de Celery para workers y tareas periódicas.
"""
from celery import Celery

from app.config import settings

celery = Celery(
    "traffic_analysis",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# Alias para mantener compatibilidad con imports existentes.
celery_app = celery


@celery.task(name="traffic.health.ping")
def ping():
    """Tarea mínima para validar que el worker está operativo."""
    return "pong"

"""
Configuración de base de datos
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from datetime import datetime
import logging

from app.config import settings
from app.models.database import Base

logger = logging.getLogger(__name__)


# Crear engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,  # Verificar conexión antes de usar
    echo=settings.DEBUG
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db():
    """Inicializar base de datos (crear tablas)"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Base de datos inicializada")
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {e}")
        raise


def get_db() -> Session:
    """
    Dependency para FastAPI
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager para uso fuera de FastAPI
    
    Usage:
        with get_db_context() as db:
            items = db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


class DatabaseManager:
    """Manager para operaciones comunes de base de datos"""
    
    @staticmethod
    def save_detection(
        db: Session,
        camera_id: str,
        person_count: int,
        detections_data: list
    ):
        """Guardar detección"""
        from app.models.database import Detection
        
        detection = Detection(
            camera_id=camera_id,
            timestamp=datetime.now(),
            person_count=person_count,
            detections_data=detections_data
        )
        
        db.add(detection)
        db.commit()
        db.refresh(detection)
        
        return detection
    
    @staticmethod
    def save_line_count(
        db: Session,
        camera_id: str,
        line_name: str,
        positive_count: int,
        negative_count: int
    ):
        """Guardar conteo de línea"""
        from app.models.database import LineCount
        
        line_count = LineCount(
            camera_id=camera_id,
            line_name=line_name,
            positive_count=positive_count,
            negative_count=negative_count,
            total_count=positive_count + negative_count
        )
        
        db.add(line_count)
        db.commit()
        db.refresh(line_count)
        
        return line_count
    
    @staticmethod
    def update_camera_status(
        db: Session,
        camera_id: str,
        **kwargs
    ):
        """Actualizar estado de cámara"""
        from app.models.database import CameraStatus
        from datetime import datetime
        
        status = db.query(CameraStatus).filter(
            CameraStatus.camera_id == camera_id
        ).first()
        
        if status is None:
            # Crear nuevo registro
            status = CameraStatus(camera_id=camera_id)
            db.add(status)
        
        # Actualizar campos
        for key, value in kwargs.items():
            if hasattr(status, key):
                setattr(status, key, value)
        
        status.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(status)
        
        return status
    
    @staticmethod
    def get_hourly_metrics(
        db: Session,
        camera_id: str,
        date: datetime,
        hour: int
    ):
        """Obtener métricas de una hora específica"""
        from app.models.database import HourlyMetrics
        
        return db.query(HourlyMetrics).filter(
            HourlyMetrics.camera_id == camera_id,
            HourlyMetrics.date == date.date(),
            HourlyMetrics.hour == hour
        ).first()
    
    @staticmethod
    def save_conversion_rate(
        db: Session,
        camera_id: str,
        local_id: str,
        date: datetime,
        visitors: int,
        transactions: int,
        revenue: float = 0.0
    ):
        """Guardar tasa de conversión"""
        from app.models.database import ConversionRate
        
        conversion_rate = (transactions / visitors * 100) if visitors > 0 else 0.0
        
        # Buscar si ya existe
        existing = db.query(ConversionRate).filter(
            ConversionRate.local_id == local_id,
            ConversionRate.date == date.date()
        ).first()
        
        if existing:
            # Actualizar
            existing.visitors = visitors
            existing.transactions = transactions
            existing.conversion_rate = conversion_rate
            existing.revenue = revenue
            result = existing
        else:
            # Crear nuevo
            result = ConversionRate(
                camera_id=camera_id,
                local_id=local_id,
                date=date.date(),
                visitors=visitors,
                transactions=transactions,
                conversion_rate=conversion_rate,
                revenue=revenue
            )
            db.add(result)
        
        db.commit()
        db.refresh(result)
        
        return result

    @staticmethod
    def save_crossing_event(
        db: Session,
        camera_id: str,
        line_name: str,
        direction: str,
        event_type: str,
        track_id: int | None = None,
        event_metadata: dict | None = None,
        timestamp: datetime | None = None
    ):
        """Guardar evento de cruce (entrada/salida)."""
        from app.models.database import CrossingEvent

        event = CrossingEvent(
            camera_id=camera_id,
            line_name=line_name,
            direction=direction,
            event_type=event_type,
            track_id=track_id,
            event_metadata=event_metadata or {},
            timestamp=timestamp or datetime.utcnow()
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def save_crossing_events(
        db: Session,
        events: list[dict]
    ):
        """Guardar múltiples eventos de cruce en una sola transacción."""
        from app.models.database import CrossingEvent

        rows = []
        for data in events:
            rows.append(
                CrossingEvent(
                    camera_id=data["camera_id"],
                    line_name=data.get("line_name", "main_gate"),
                    direction=data["direction"],
                    event_type=data["event_type"],
                    track_id=data.get("track_id"),
                    event_metadata=data.get("event_metadata", {}),
                    timestamp=data.get("timestamp", datetime.utcnow())
                )
            )
        db.add_all(rows)
        db.commit()
        return rows

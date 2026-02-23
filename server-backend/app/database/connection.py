"""
Configuración de base de datos
"""
from sqlalchemy import create_engine, func, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from datetime import datetime, date, time, timedelta
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
        _ensure_hourly_metrics_columns()
        logger.info("✓ Base de datos inicializada")
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {e}")
        raise


def _ensure_hourly_metrics_columns():
    """
    Asegura columnas nuevas en tablas existentes sin requerir migraciones manuales.
    """
    try:
        inspector = inspect(engine)
        if "hourly_metrics" not in inspector.get_table_names():
            return

        columns = {col["name"] for col in inspector.get_columns("hourly_metrics")}
        if "local_id" not in columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE hourly_metrics ADD COLUMN local_id VARCHAR(50) NULL"))
            logger.info("✓ Columna hourly_metrics.local_id creada")
    except Exception as exc:
        # No bloqueamos startup si la columna ya existe o el motor responde distinto.
        logger.warning(f"No se pudo validar/crear columnas en hourly_metrics: {exc}")


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

    @staticmethod
    def aggregate_completed_hourly_metrics(db: Session) -> dict:
        """
        Agrega detecciones por hora (horas cerradas), guarda en hourly_metrics y
        elimina detections utilizadas para reducir volumen.
        """
        from app.models.database import CameraStatus, CrossingEvent, Detection, HourlyMetrics

        current_hour_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        grouped = db.query(
            Detection.camera_id.label("camera_id"),
            func.date(Detection.timestamp).label("target_date"),
            func.extract("hour", Detection.timestamp).label("target_hour"),
            func.avg(Detection.person_count).label("avg_count"),
            func.max(Detection.person_count).label("peak_count"),
            func.count(Detection.id).label("sample_count"),
        ).filter(
            Detection.timestamp < current_hour_start
        ).group_by(
            Detection.camera_id,
            func.date(Detection.timestamp),
            func.extract("hour", Detection.timestamp)
        ).all()

        if not grouped:
            return {"processed_hours": 0, "deleted_detections": 0}

        camera_ids = list({row.camera_id for row in grouped})
        status_rows = db.query(CameraStatus).filter(CameraStatus.camera_id.in_(camera_ids)).all()
        local_by_camera: dict[str, str | None] = {}
        for row in status_rows:
            metadata = row.camera_metadata if isinstance(row.camera_metadata, dict) else {}
            local_by_camera[row.camera_id] = metadata.get("branch_id") or metadata.get("local_id")

        processed_hours = 0
        deleted_detections = 0

        for row in grouped:
            camera_id = row.camera_id
            target_date = row.target_date if isinstance(row.target_date, date) else datetime.utcnow().date()
            target_hour = int(row.target_hour)
            avg_count = float(row.avg_count or 0.0)
            peak_count = int(row.peak_count or 0)

            hour_start = datetime.combine(target_date, time(hour=target_hour))
            hour_end = hour_start + timedelta(hours=1)

            crossings = db.query(
                CrossingEvent.event_type,
                func.count(CrossingEvent.id).label("cnt")
            ).filter(
                CrossingEvent.camera_id == camera_id,
                CrossingEvent.timestamp >= hour_start,
                CrossingEvent.timestamp < hour_end
            ).group_by(
                CrossingEvent.event_type
            ).all()

            entries = 0
            for c in crossings:
                if c.event_type == "entry":
                    entries = int(c.cnt)

            hourly = db.query(HourlyMetrics).filter(
                HourlyMetrics.camera_id == camera_id,
                HourlyMetrics.date == target_date,
                HourlyMetrics.hour == target_hour
            ).first()

            if hourly is None:
                hourly = HourlyMetrics(
                    camera_id=camera_id,
                    local_id=local_by_camera.get(camera_id),
                    date=target_date,
                    hour=target_hour,
                    total_visitors=entries,
                    peak_count=peak_count,
                    avg_dwell_time=avg_count,  # promedio de ocupación
                )
                db.add(hourly)
            else:
                hourly.local_id = local_by_camera.get(camera_id)
                hourly.total_visitors = entries
                hourly.peak_count = peak_count
                hourly.avg_dwell_time = avg_count

            deleted = db.query(Detection).filter(
                Detection.camera_id == camera_id,
                Detection.timestamp >= hour_start,
                Detection.timestamp < hour_end
            ).delete(synchronize_session=False)
            deleted_detections += int(deleted or 0)
            processed_hours += 1

        db.commit()
        return {
            "processed_hours": processed_hours,
            "deleted_detections": deleted_detections
        }

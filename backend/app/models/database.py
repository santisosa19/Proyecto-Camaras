"""
Modelos de base de datos
"""
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, JSON, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Detection(Base):
    """Tabla de detecciones individuales"""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    person_count = Column(Integer, default=0)
    detections_data = Column(JSON)  # Lista de detecciones con bbox, conf, etc.
    
    __table_args__ = (
        Index('idx_camera_timestamp', 'camera_id', 'timestamp'),
    )


class HourlyMetrics(Base):
    """Métricas agregadas por hora"""
    __tablename__ = "hourly_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    hour = Column(Integer, nullable=False)
    total_visitors = Column(Integer, default=0)
    peak_count = Column(Integer, default=0)
    avg_dwell_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_camera_date_hour', 'camera_id', 'date', 'hour', unique=True),
    )


class LineCount(Base):
    """Conteos de líneas virtuales"""
    __tablename__ = "line_counts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), nullable=False)
    line_name = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    total_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_line_camera_time', 'camera_id', 'line_name', 'timestamp'),
    )


class Heatmap(Base):
    """Heatmaps generados"""
    __tablename__ = "heatmaps"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    hour = Column(Integer, nullable=False)
    heatmap_path = Column(String(255))  # Ruta al archivo
    heatmap_data = Column(JSON)  # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_heatmap_camera_date', 'camera_id', 'date', 'hour', unique=True),
    )


class ConversionRate(Base):
    """Tasas de conversión (integración con Cegid)"""
    __tablename__ = "conversion_rates"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), nullable=False)
    local_id = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    visitors = Column(Integer, default=0)
    transactions = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)
    revenue = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_conversion_local_date', 'local_id', 'date', unique=True),
    )


class CameraStatus(Base):
    """Estado de las cámaras"""
    __tablename__ = "camera_status"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), unique=True, nullable=False)
    camera_name = Column(String(100))
    rtsp_url = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_connected = Column(Boolean, default=False)
    last_frame_at = Column(DateTime)
    fps = Column(Float, default=0.0)
    error_count = Column(Integer, default=0)
    total_frames = Column(Integer, default=0)
    uptime_seconds = Column(Float, default=0.0)
    camera_metadata = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CrossingEvent(Base):
    """Eventos de cruce por línea (entrada/salida)"""
    __tablename__ = "crossing_events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    line_name = Column(String(100), nullable=False, default="main_gate")
    direction = Column(String(20), nullable=False)  # positive | negative
    event_type = Column(String(20), nullable=False)  # entry | exit
    track_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index('idx_crossing_camera_time', 'camera_id', 'timestamp'),
        Index('idx_crossing_camera_type_time', 'camera_id', 'event_type', 'timestamp'),
    )

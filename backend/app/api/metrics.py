"""
API Router - Métricas
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, date, timedelta

from app.database.connection import get_db
from app.models.database import Detection, HourlyMetrics, LineCount, ConversionRate, CrossingEvent

router = APIRouter()


class RealtimeMetrics(BaseModel):
    """Métricas en tiempo real"""
    camera_id: str
    timestamp: datetime
    current_count: int
    fps: float


class HourlyStats(BaseModel):
    """Estadísticas por hora"""
    hour: int
    total_visitors: int
    peak_count: int
    avg_dwell_time: float


class DailyStats(BaseModel):
    """Estadísticas diarias"""
    date: date
    total_visitors: int
    peak_hour: int
    avg_visitors_per_hour: float


@router.get("/realtime/{camera_id}")
async def get_realtime_metrics(
    camera_id: str,
    db: Session = Depends(get_db)
):
    """
    Obtener métricas en tiempo real de una cámara
    """
    # Obtener última detección
    latest = db.query(Detection).filter(
        Detection.camera_id == camera_id
    ).order_by(Detection.timestamp.desc()).first()
    
    if not latest:
        return {
            "camera_id": camera_id,
            "timestamp": datetime.now(),
            "current_count": 0,
            "message": "No hay datos disponibles"
        }
    
    return {
        "camera_id": camera_id,
        "timestamp": latest.timestamp,
        "current_count": latest.person_count,
        "detections": latest.detections_data
    }


@router.get("/hourly/{camera_id}")
async def get_hourly_metrics(
    camera_id: str,
    target_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    Obtener métricas por hora de un día específico
    """
    if target_date is None:
        target_date = date.today()
    
    metrics = db.query(HourlyMetrics).filter(
        HourlyMetrics.camera_id == camera_id,
        HourlyMetrics.date == target_date
    ).order_by(HourlyMetrics.hour).all()
    
    return {
        "camera_id": camera_id,
        "date": target_date,
        "metrics": [
            {
                "hour": m.hour,
                "total_visitors": m.total_visitors,
                "peak_count": m.peak_count,
                "avg_dwell_time": m.avg_dwell_time
            }
            for m in metrics
        ]
    }


@router.get("/daily/{camera_id}")
async def get_daily_summary(
    camera_id: str,
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    Resumen diario de métricas
    """
    if end_date is None:
        end_date = date.today()
    
    if start_date is None:
        start_date = end_date - timedelta(days=7)
    
    # Agregar por día
    daily_stats = db.query(
        func.date(HourlyMetrics.date).label('date'),
        func.sum(HourlyMetrics.total_visitors).label('total_visitors'),
        func.max(HourlyMetrics.peak_count).label('peak_count'),
        func.avg(HourlyMetrics.total_visitors).label('avg_per_hour')
    ).filter(
        HourlyMetrics.camera_id == camera_id,
        HourlyMetrics.date >= start_date,
        HourlyMetrics.date <= end_date
    ).group_by(
        func.date(HourlyMetrics.date)
    ).all()
    
    return {
        "camera_id": camera_id,
        "period": {
            "start": start_date,
            "end": end_date
        },
        "daily_stats": [
            {
                "date": stat.date,
                "total_visitors": stat.total_visitors or 0,
                "peak_count": stat.peak_count or 0,
                "avg_per_hour": round(stat.avg_per_hour or 0, 2)
            }
            for stat in daily_stats
        ]
    }


@router.get("/lines/{camera_id}")
async def get_line_counts(
    camera_id: str,
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    Obtener conteos de líneas
    """
    if end_time is None:
        end_time = datetime.now()
    
    if start_time is None:
        start_time = end_time - timedelta(hours=1)
    
    line_counts = db.query(LineCount).filter(
        LineCount.camera_id == camera_id,
        LineCount.timestamp >= start_time,
        LineCount.timestamp <= end_time
    ).all()
    
    # Agrupar por línea
    lines_summary = {}
    for lc in line_counts:
        if lc.line_name not in lines_summary:
            lines_summary[lc.line_name] = {
                'positive': 0,
                'negative': 0,
                'total': 0
            }
        
        lines_summary[lc.line_name]['positive'] += lc.positive_count
        lines_summary[lc.line_name]['negative'] += lc.negative_count
        lines_summary[lc.line_name]['total'] += lc.total_count
    
    return {
        "camera_id": camera_id,
        "period": {
            "start": start_time,
            "end": end_time
        },
        "lines": lines_summary
    }


@router.get("/conversion/{local_id}")
async def get_conversion_rate(
    local_id: str,
    target_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    Obtener tasa de conversión de un local
    """
    if target_date is None:
        target_date = date.today()
    
    conversion = db.query(ConversionRate).filter(
        ConversionRate.local_id == local_id,
        ConversionRate.date == target_date
    ).first()
    
    if not conversion:
        return {
            "local_id": local_id,
            "date": target_date,
            "message": "No hay datos de conversión disponibles",
            "conversion_rate": 0
        }
    
    return {
        "local_id": local_id,
        "date": target_date,
        "visitors": conversion.visitors,
        "transactions": conversion.transactions,
        "conversion_rate": conversion.conversion_rate,
        "revenue": conversion.revenue
    }


@router.get("/comparison")
async def compare_cameras(
    camera_ids: List[str] = Query(...),
    target_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    Comparar métricas entre múltiples cámaras
    """
    if target_date is None:
        target_date = date.today()
    
    comparison = []
    
    for camera_id in camera_ids:
        # Obtener total de visitantes del día
        total = db.query(
            func.sum(HourlyMetrics.total_visitors)
        ).filter(
            HourlyMetrics.camera_id == camera_id,
            HourlyMetrics.date == target_date
        ).scalar()
        
        # Obtener hora pico
        peak_hour = db.query(
            HourlyMetrics.hour,
            func.max(HourlyMetrics.peak_count).label('peak')
        ).filter(
            HourlyMetrics.camera_id == camera_id,
            HourlyMetrics.date == target_date
        ).group_by(HourlyMetrics.hour).order_by(
            func.max(HourlyMetrics.peak_count).desc()
        ).first()
        
        comparison.append({
            "camera_id": camera_id,
            "total_visitors": total or 0,
            "peak_hour": peak_hour.hour if peak_hour else None,
            "peak_count": peak_hour.peak if peak_hour else 0
        })
    
    return {
        "date": target_date,
        "cameras": comparison
    }


@router.get("/flow/{camera_id}")
async def get_hourly_flow(
    camera_id: str,
    target_date: Optional[date] = Query(default=None),
    initial_occupancy: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Flujo horario basado en eventos de cruce:
    - entries: personas que entraron en esa hora
    - exits: personas que salieron en esa hora
    - occupancy_end: aforo acumulado al cierre de esa hora
    """
    if target_date is None:
        target_date = date.today()

    grouped = db.query(
        func.extract('hour', CrossingEvent.timestamp).label('hour'),
        CrossingEvent.event_type.label('event_type'),
        func.count(CrossingEvent.id).label('count')
    ).filter(
        CrossingEvent.camera_id == camera_id,
        func.date(CrossingEvent.timestamp) == target_date
    ).group_by(
        func.extract('hour', CrossingEvent.timestamp),
        CrossingEvent.event_type
    ).all()

    hourly = {
        hour: {
            "hour": hour,
            "entries": 0,
            "exits": 0,
            "net_flow": 0,
            "occupancy_end": 0
        }
        for hour in range(24)
    }

    for row in grouped:
        hour = int(row.hour)
        if row.event_type == "entry":
            hourly[hour]["entries"] = int(row.count)
        elif row.event_type == "exit":
            hourly[hour]["exits"] = int(row.count)

    occupancy = initial_occupancy
    for hour in range(24):
        entries = hourly[hour]["entries"]
        exits = hourly[hour]["exits"]
        net = entries - exits
        occupancy = max(0, occupancy + net)
        hourly[hour]["net_flow"] = net
        hourly[hour]["occupancy_end"] = occupancy

    totals = {
        "entries": sum(h["entries"] for h in hourly.values()),
        "exits": sum(h["exits"] for h in hourly.values()),
        "net_flow": sum(h["net_flow"] for h in hourly.values()),
        "occupancy_end_of_day": occupancy
    }

    return {
        "camera_id": camera_id,
        "date": target_date,
        "initial_occupancy": initial_occupancy,
        "totals": totals,
        "hours": [hourly[h] for h in range(24)]
    }

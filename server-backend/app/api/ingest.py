"""
API Router - Ingesta remota (sucursales -> servidor central)
"""
from datetime import datetime, time
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import get_db, DatabaseManager
from app.models.database import CameraStatus, Detection, Heatmap

router = APIRouter()


def _authorize_ingest(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    """
    Valida API key si está configurada en servidor central.
    Si INGEST_API_KEY está vacía, deja acceso abierto para entorno local.
    """
    required_key = settings.INGEST_API_KEY.strip()
    if not required_key:
        return

    if x_api_key != required_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida para endpoint de ingesta"
        )


class CrossingIngestItem(BaseModel):
    camera_id: str
    line_name: str = "main_gate"
    direction: Literal["positive", "negative"]
    event_type: Literal["entry", "exit"]
    track_id: int | None = None
    timestamp: datetime | None = None
    event_metadata: dict[str, Any] = Field(default_factory=dict)


class CrossingIngestRequest(BaseModel):
    events: list[CrossingIngestItem]


class DetectionSnapshotItem(BaseModel):
    camera_id: str
    camera_name: str | None = None
    rtsp_url: str | None = None
    branch_id: str | None = None
    branch_name: str | None = None
    timestamp: datetime | None = None
    person_count: int = 0
    detections_data: list[dict[str, Any]] = Field(default_factory=list)
    is_connected: bool | None = None
    fps: float | None = None
    total_frames: int | None = None
    error_count: int | None = None


class DetectionSnapshotRequest(BaseModel):
    items: list[DetectionSnapshotItem]


class HeatmapIngestItem(BaseModel):
    camera_id: str
    camera_name: str | None = None
    branch_id: str | None = None
    branch_name: str | None = None
    hour_start: datetime
    generated_at: datetime | None = None
    frame_width: int
    frame_height: int
    cell_size: int
    grid: list[list[float]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    background_image_base64: str | None = None
    overlay_png_base64: str | None = None
    is_partial: bool = False


class HeatmapIngestRequest(BaseModel):
    items: list[HeatmapIngestItem]


@router.post("/crossings")
async def ingest_crossings(
    payload: CrossingIngestRequest,
    _: None = Depends(_authorize_ingest),
    db: Session = Depends(get_db)
):
    """
    Ingestar eventos de cruce (entry/exit) en lote.
    """
    if not payload.events:
        return {"inserted": 0}

    try:
        rows = [item.model_dump() for item in payload.events]
        for row in rows:
            if row.get("timestamp") is None:
                row["timestamp"] = datetime.utcnow()

        saved = DatabaseManager.save_crossing_events(db=db, events=rows)
        return {"inserted": len(saved)}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error guardando crossings: {exc}")


@router.post("/detections")
async def ingest_detections(
    payload: DetectionSnapshotRequest,
    _: None = Depends(_authorize_ingest),
    db: Session = Depends(get_db)
):
    """
    Ingestar snapshots de detección en lote.
    """
    if not payload.items:
        return {"inserted": 0, "updated_cameras": 0}

    try:
        detection_rows: list[Detection] = []
        latest_by_camera: dict[str, DetectionSnapshotItem] = {}

        for item in payload.items:
            if item.person_count <= 0:
                continue
            ts = item.timestamp or datetime.utcnow()
            detection_rows.append(
                Detection(
                    camera_id=item.camera_id,
                    timestamp=ts,
                    person_count=item.person_count,
                    detections_data=item.detections_data,
                )
            )

            prev = latest_by_camera.get(item.camera_id)
            if prev is None:
                latest_by_camera[item.camera_id] = item
            else:
                prev_ts = prev.timestamp or datetime.min
                if ts >= prev_ts:
                    latest_by_camera[item.camera_id] = item

        if not detection_rows:
            return {"inserted": 0, "updated_cameras": 0}

        db.add_all(detection_rows)

        updated_cameras = 0
        for camera_id, item in latest_by_camera.items():
            status_row = db.query(CameraStatus).filter(
                CameraStatus.camera_id == camera_id
            ).first()

            if status_row is None:
                status_row = CameraStatus(camera_id=camera_id)
                db.add(status_row)

            if item.camera_name:
                status_row.camera_name = item.camera_name
            if item.rtsp_url:
                status_row.rtsp_url = item.rtsp_url
            if item.is_connected is not None:
                status_row.is_connected = item.is_connected
            if item.fps is not None:
                status_row.fps = item.fps
            if item.total_frames is not None:
                status_row.total_frames = item.total_frames
            if item.error_count is not None:
                status_row.error_count = item.error_count
            existing_meta = status_row.camera_metadata if isinstance(status_row.camera_metadata, dict) else {}
            if item.branch_id:
                existing_meta["branch_id"] = item.branch_id
            if item.branch_name:
                existing_meta["branch_name"] = item.branch_name
            if existing_meta:
                status_row.camera_metadata = existing_meta
            status_row.last_frame_at = item.timestamp or datetime.utcnow()
            status_row.updated_at = datetime.utcnow()
            status_row.is_active = True
            updated_cameras += 1

        db.commit()
        return {"inserted": len(detection_rows), "updated_cameras": updated_cameras}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error guardando detections: {exc}")


@router.post("/heatmaps")
async def ingest_heatmaps(
    payload: HeatmapIngestRequest,
    _: None = Depends(_authorize_ingest),
    db: Session = Depends(get_db)
):
    """
    Ingestar heatmaps por hora en lote (uno por cámara/hora).
    """
    if not payload.items:
        return {"upserted": 0, "updated_cameras": 0}

    try:
        upserted = 0
        updated_cameras = 0

        for item in payload.items:
            hour_start = item.hour_start.replace(minute=0, second=0, microsecond=0)
            day_start = datetime.combine(hour_start.date(), time.min)
            hour = int(hour_start.hour)

            row = db.query(Heatmap).filter(
                Heatmap.camera_id == item.camera_id,
                Heatmap.date == day_start,
                Heatmap.hour == hour,
            ).first()

            heatmap_data = {
                "hour_start": hour_start.isoformat(),
                "generated_at": (item.generated_at or datetime.utcnow()).isoformat(),
                "frame_width": item.frame_width,
                "frame_height": item.frame_height,
                "cell_size": item.cell_size,
                "grid": item.grid,
                "stats": item.stats,
                "branch_id": item.branch_id,
                "branch_name": item.branch_name,
                "background_image_base64": item.background_image_base64,
                "overlay_png_base64": item.overlay_png_base64,
                "is_partial": bool(item.is_partial),
            }

            if row is None:
                row = Heatmap(
                    camera_id=item.camera_id,
                    date=day_start,
                    hour=hour,
                    heatmap_data=heatmap_data,
                )
                db.add(row)
            else:
                row.heatmap_data = heatmap_data

            status_row = db.query(CameraStatus).filter(
                CameraStatus.camera_id == item.camera_id
            ).first()
            if status_row is None:
                status_row = CameraStatus(camera_id=item.camera_id)
                db.add(status_row)

            if item.camera_name:
                status_row.camera_name = item.camera_name
            existing_meta = status_row.camera_metadata if isinstance(status_row.camera_metadata, dict) else {}
            if item.branch_id:
                existing_meta["branch_id"] = item.branch_id
            if item.branch_name:
                existing_meta["branch_name"] = item.branch_name
            if existing_meta:
                status_row.camera_metadata = existing_meta
            status_row.updated_at = datetime.utcnow()
            status_row.is_active = True

            upserted += 1
            updated_cameras += 1

        db.commit()
        return {"upserted": upserted, "updated_cameras": updated_cameras}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error guardando heatmaps: {exc}")

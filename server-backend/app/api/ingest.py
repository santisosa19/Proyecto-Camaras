"""
API Router - Ingesta remota (sucursales -> servidor central)
"""
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import get_db, DatabaseManager
from app.models.database import CameraStatus, Detection

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
    timestamp: datetime | None = None
    person_count: int = 0
    detections_data: list[dict[str, Any]] = Field(default_factory=list)
    is_connected: bool | None = None
    fps: float | None = None
    total_frames: int | None = None
    error_count: int | None = None


class DetectionSnapshotRequest(BaseModel):
    items: list[DetectionSnapshotItem]


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
            status_row.last_frame_at = item.timestamp or datetime.utcnow()
            status_row.updated_at = datetime.utcnow()
            status_row.is_active = True
            updated_cameras += 1

        db.commit()
        return {"inserted": len(detection_rows), "updated_cameras": updated_cameras}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error guardando detections: {exc}")

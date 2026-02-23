"""
API Router - Dashboard multi-sucursal
"""
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models.database import CameraStatus, CrossingEvent, Detection

router = APIRouter()


def _camera_branch_info(camera: CameraStatus) -> tuple[str, str]:
    metadata = camera.camera_metadata if isinstance(camera.camera_metadata, dict) else {}
    branch_id = (
        metadata.get("branch_id")
        or metadata.get("local_id")
        or "default_branch"
    )
    branch_name = metadata.get("branch_name") or metadata.get("local_name") or branch_id
    return branch_id, branch_name


def _latest_counts_by_camera(db: Session, camera_ids: list[str]) -> dict[str, int]:
    if not camera_ids:
        return {}

    result: dict[str, int] = {}
    for camera_id in camera_ids:
        latest = db.query(Detection).filter(
            Detection.camera_id == camera_id
        ).order_by(Detection.timestamp.desc()).first()
        result[camera_id] = latest.person_count if latest else 0
    return result


def _count_crossings_today(db: Session, camera_ids: list[str]) -> tuple[int, int]:
    if not camera_ids:
        return 0, 0

    start = datetime.combine(date.today(), datetime.min.time())
    end = start + timedelta(days=1)
    rows = db.query(CrossingEvent).filter(
        CrossingEvent.camera_id.in_(camera_ids),
        CrossingEvent.timestamp >= start,
        CrossingEvent.timestamp < end
    ).all()

    entries = sum(1 for r in rows if r.event_type == "entry")
    exits = sum(1 for r in rows if r.event_type == "exit")
    return entries, exits


@router.get("/overview")
async def get_overview(db: Session = Depends(get_db)):
    """
    Resumen global para página principal del dashboard.
    """
    cameras = db.query(CameraStatus).all()
    camera_ids = [c.camera_id for c in cameras]

    latest_counts = _latest_counts_by_camera(db, camera_ids)
    entries_today, exits_today = _count_crossings_today(db, camera_ids)

    branches: dict[str, str] = {}
    for c in cameras:
        branch_id, branch_name = _camera_branch_info(c)
        branches[branch_id] = branch_name

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_branches": len(branches),
        "total_cameras": len(cameras),
        "online_cameras": sum(1 for c in cameras if c.is_connected),
        "current_occupancy": sum(latest_counts.values()),
        "entries_today": entries_today,
        "exits_today": exits_today,
    }


@router.get("/branches")
async def list_branch_cards(db: Session = Depends(get_db)):
    """
    Tarjetas resumen por sucursal para vista global.
    """
    cameras = db.query(CameraStatus).all()
    camera_ids = [c.camera_id for c in cameras]
    latest_counts = _latest_counts_by_camera(db, camera_ids)
    entries_today, exits_today = _count_crossings_today(db, camera_ids)

    by_branch: dict[str, dict[str, Any]] = {}
    for c in cameras:
        branch_id, branch_name = _camera_branch_info(c)
        card = by_branch.setdefault(
            branch_id,
            {
                "branch_id": branch_id,
                "branch_name": branch_name,
                "total_cameras": 0,
                "online_cameras": 0,
                "current_occupancy": 0,
                "camera_ids": [],
            }
        )
        card["total_cameras"] += 1
        if c.is_connected:
            card["online_cameras"] += 1
        card["current_occupancy"] += latest_counts.get(c.camera_id, 0)
        card["camera_ids"].append(c.camera_id)

    # Recalcular entradas/salidas por sucursal
    for card in by_branch.values():
        cam_ids = card.pop("camera_ids")
        b_entries, b_exits = _count_crossings_today(db, cam_ids)
        card["entries_today"] = b_entries
        card["exits_today"] = b_exits

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "entries_today": entries_today,
            "exits_today": exits_today,
        },
        "branches": sorted(by_branch.values(), key=lambda x: x["branch_name"]),
    }


@router.get("/branches/{branch_id}")
async def get_branch_detail(
    branch_id: str,
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db)
):
    """
    Detalle de sucursal para dashboard.
    """
    cameras = db.query(CameraStatus).all()
    branch_cameras = [c for c in cameras if _camera_branch_info(c)[0] == branch_id]
    camera_ids = [c.camera_id for c in branch_cameras]

    latest_counts = _latest_counts_by_camera(db, camera_ids)
    entries_today, exits_today = _count_crossings_today(db, camera_ids)

    since = datetime.utcnow() - timedelta(hours=hours)
    rows = db.query(CrossingEvent).filter(
        CrossingEvent.camera_id.in_(camera_ids),
        CrossingEvent.timestamp >= since
    ).all()

    hourly: dict[str, dict[str, int]] = {}
    for row in rows:
        key = row.timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        bucket = hourly.setdefault(key, {"entry": 0, "exit": 0})
        if row.event_type == "entry":
            bucket["entry"] += 1
        elif row.event_type == "exit":
            bucket["exit"] += 1

    cameras_payload = []
    for c in branch_cameras:
        cameras_payload.append(
            {
                "camera_id": c.camera_id,
                "camera_name": c.camera_name,
                "is_connected": c.is_connected,
                "fps": c.fps,
                "error_count": c.error_count,
                "last_frame_at": c.last_frame_at,
                "current_count": latest_counts.get(c.camera_id, 0),
            }
        )

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "branch_id": branch_id,
        "branch_name": _camera_branch_info(branch_cameras[0])[1] if branch_cameras else branch_id,
        "entries_today": entries_today,
        "exits_today": exits_today,
        "current_occupancy": sum(latest_counts.values()),
        "cameras": cameras_payload,
        "hourly_flow": [
            {"hour": k, "entry": v["entry"], "exit": v["exit"]}
            for k, v in sorted(hourly.items())
        ],
    }

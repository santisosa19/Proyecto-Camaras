"""
API Router - Dashboard multi-sucursal
"""
from collections import defaultdict
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models.database import CameraStatus, CrossingEvent, Detection, Heatmap
from app.security import require_authenticated_user

router = APIRouter(dependencies=[Depends(require_authenticated_user)])


def _utcnow() -> datetime:
    return datetime.utcnow()


def _resolve_time_window(
    hours: int,
    start_date: date | None,
    end_date: date | None,
) -> tuple[datetime, datetime, date, date]:
    if start_date is None and end_date is None:
        since = _utcnow() - timedelta(hours=hours)
        until = _utcnow()
        return since, until, since.date(), until.date()

    if start_date is None:
        start_date = end_date
    if end_date is None:
        end_date = start_date

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    since = datetime.combine(start_date, datetime.min.time())
    until = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
    return since, until, start_date, end_date


def _empty_gender_bucket() -> dict[str, int]:
    return {"male": 0, "female": 0, "unknown": 0}


def _normalize_apparent_gender(value: Any) -> str:
    raw = (str(value).strip().lower() if value is not None else "")
    if raw in {"male", "m", "man", "masculino", "hombre"}:
        return "male"
    if raw in {"female", "f", "woman", "femenino", "mujer"}:
        return "female"
    return "unknown"


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

    result: dict[str, int] = {camera_id: 0 for camera_id in camera_ids}

    latest_ts_subquery = (
        db.query(
            Detection.camera_id.label("camera_id"),
            func.max(Detection.timestamp).label("max_ts"),
        )
        .filter(Detection.camera_id.in_(camera_ids))
        .group_by(Detection.camera_id)
        .subquery()
    )

    rows = (
        db.query(
            Detection.camera_id,
            func.max(Detection.person_count).label("person_count"),
        )
        .join(
            latest_ts_subquery,
            and_(
                Detection.camera_id == latest_ts_subquery.c.camera_id,
                Detection.timestamp == latest_ts_subquery.c.max_ts,
            ),
        )
        .group_by(Detection.camera_id)
        .all()
    )
    for row in rows:
        result[row.camera_id] = int(row.person_count or 0)
    return result


def _crossings_in_range_by_camera(
    db: Session,
    camera_ids: list[str],
    since: datetime,
    until: datetime,
) -> dict[str, dict[str, int]]:
    if not camera_ids:
        return {}

    rows = (
        db.query(
            CrossingEvent.camera_id,
            CrossingEvent.event_type,
            func.count(CrossingEvent.id).label("count"),
        )
        .filter(
            CrossingEvent.camera_id.in_(camera_ids),
            CrossingEvent.timestamp >= since,
            CrossingEvent.timestamp < until,
        )
        .group_by(CrossingEvent.camera_id, CrossingEvent.event_type)
        .all()
    )

    result: dict[str, dict[str, int]] = {camera_id: {"entry": 0, "exit": 0} for camera_id in camera_ids}
    for row in rows:
        bucket = result.setdefault(row.camera_id, {"entry": 0, "exit": 0})
        if row.event_type in bucket:
            bucket[row.event_type] = int(row.count or 0)
    return result


def _crossings_in_range_with_gender_by_camera(
    db: Session,
    camera_ids: list[str],
    since: datetime,
    until: datetime,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    if not camera_ids:
        return {}, {}

    counts = {camera_id: {"entry": 0, "exit": 0} for camera_id in camera_ids}
    gender_counts = {
        camera_id: {"entry": _empty_gender_bucket(), "exit": _empty_gender_bucket()}
        for camera_id in camera_ids
    }

    rows = (
        db.query(
            CrossingEvent.camera_id,
            CrossingEvent.event_type,
            CrossingEvent.event_metadata,
        )
        .filter(
            CrossingEvent.camera_id.in_(camera_ids),
            CrossingEvent.timestamp >= since,
            CrossingEvent.timestamp < until,
        )
        .all()
    )

    for row in rows:
        if row.event_type not in {"entry", "exit"}:
            continue
        counts[row.camera_id][row.event_type] += 1

        metadata = row.event_metadata if isinstance(row.event_metadata, dict) else {}
        gender = _normalize_apparent_gender(metadata.get("apparent_gender"))
        gender_counts[row.camera_id][row.event_type][gender] += 1

    return counts, gender_counts


def _sum_gender_counts(
    gender_counts_by_camera: dict[str, dict[str, dict[str, int]]],
    event_type: Literal["entry", "exit"],
) -> dict[str, int]:
    total = _empty_gender_bucket()
    for camera_counts in gender_counts_by_camera.values():
        bucket = camera_counts.get(event_type) or {}
        total["male"] += int(bucket.get("male", 0))
        total["female"] += int(bucket.get("female", 0))
        total["unknown"] += int(bucket.get("unknown", 0))
    return total


def _hourly_flow(
    db: Session,
    camera_ids: list[str],
    since: datetime,
    until: datetime,
) -> list[dict[str, Any]]:
    if not camera_ids:
        return []

    if until <= since:
        return []

    rows = (
        db.query(CrossingEvent.timestamp, CrossingEvent.event_type)
        .filter(
            CrossingEvent.camera_id.in_(camera_ids),
            CrossingEvent.timestamp >= since,
            CrossingEvent.timestamp < until,
        )
        .all()
    )

    buckets: dict[datetime, dict[str, int]] = defaultdict(lambda: {"entry": 0, "exit": 0})
    for row in rows:
        bucket_dt = row.timestamp.replace(minute=0, second=0, microsecond=0)
        if row.event_type == "entry":
            buckets[bucket_dt]["entry"] += 1
        elif row.event_type == "exit":
            buckets[bucket_dt]["exit"] += 1

    end_hour = (until - timedelta(seconds=1)).replace(minute=0, second=0, microsecond=0)
    start = since.replace(minute=0, second=0, microsecond=0)

    occupancy = 0
    flow_series: list[dict[str, Any]] = []
    current = start
    while current <= end_hour:
        entry = buckets[current]["entry"] if current in buckets else 0
        exit_ = buckets[current]["exit"] if current in buckets else 0
        net = entry - exit_
        occupancy += net
        flow_series.append(
            {
                "hour": current.isoformat(),
                "entry": entry,
                "exit": exit_,
                "net": net,
                "occupancy_end": max(0, occupancy),
            }
        )
        current += timedelta(hours=1)
    return flow_series


def _build_branch_cards(
    cameras: list[CameraStatus],
    latest_counts: dict[str, int],
    crossings_by_camera: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    by_branch: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        branch_id, branch_name = _camera_branch_info(camera)
        card = by_branch.setdefault(
            branch_id,
            {
                "branch_id": branch_id,
                "branch_name": branch_name,
                "total_cameras": 0,
                "online_cameras": 0,
                "current_occupancy": 0,
                "entries_today": 0,
                "exits_today": 0,
                "net_today": 0,
                "online_ratio": 0.0,
            },
        )

        card["total_cameras"] += 1
        if camera.is_connected:
            card["online_cameras"] += 1
        card["current_occupancy"] += latest_counts.get(camera.camera_id, 0)
        card["entries_today"] += crossings_by_camera.get(camera.camera_id, {}).get("entry", 0)
        card["exits_today"] += crossings_by_camera.get(camera.camera_id, {}).get("exit", 0)

    for card in by_branch.values():
        card["net_today"] = card["entries_today"] - card["exits_today"]
        total = max(1, card["total_cameras"])
        card["online_ratio"] = round(card["online_cameras"] / total, 3)

    return sorted(by_branch.values(), key=lambda item: item["branch_name"])


def _build_alerts(cameras: list[CameraStatus], latest_counts: dict[str, int]) -> list[dict[str, Any]]:
    now = _utcnow()
    alerts: list[dict[str, Any]] = []

    for camera in cameras:
        branch_id, branch_name = _camera_branch_info(camera)
        if not camera.is_connected:
            alerts.append(
                {
                    "severity": "high",
                    "title": "Cámara fuera de línea",
                    "description": f"{camera.camera_name or camera.camera_id} no está conectada",
                    "camera_id": camera.camera_id,
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                }
            )
            continue

        if (camera.error_count or 0) >= 10:
            alerts.append(
                {
                    "severity": "medium",
                    "title": "Errores elevados en cámara",
                    "description": (
                        f"{camera.camera_name or camera.camera_id} acumula {camera.error_count} errores"
                    ),
                    "camera_id": camera.camera_id,
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                }
            )

        if camera.last_frame_at is not None:
            age = now - camera.last_frame_at
            if age > timedelta(minutes=5):
                alerts.append(
                    {
                        "severity": "medium",
                        "title": "Frame desactualizado",
                        "description": (
                            f"{camera.camera_name or camera.camera_id} no reporta frames hace "
                            f"{int(age.total_seconds() // 60)} min"
                        ),
                        "camera_id": camera.camera_id,
                        "branch_id": branch_id,
                        "branch_name": branch_name,
                    }
                )

        if latest_counts.get(camera.camera_id, 0) >= 50:
            alerts.append(
                {
                    "severity": "low",
                    "title": "Alta ocupación detectada",
                    "description": (
                        f"{camera.camera_name or camera.camera_id} reporta "
                        f"{latest_counts.get(camera.camera_id, 0)} personas"
                    ),
                    "camera_id": camera.camera_id,
                    "branch_id": branch_id,
                    "branch_name": branch_name,
                }
            )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda item: severity_order.get(item["severity"], 99))
    return alerts


def _sort_branches(
    branches: list[dict[str, Any]],
    sort_by: Literal["name", "occupancy", "entries", "exits", "online_ratio"],
    order: Literal["asc", "desc"],
) -> list[dict[str, Any]]:
    reverse = order == "desc"
    if sort_by == "name":
        return sorted(branches, key=lambda item: item["branch_name"], reverse=reverse)
    if sort_by == "occupancy":
        return sorted(branches, key=lambda item: item["current_occupancy"], reverse=reverse)
    if sort_by == "entries":
        return sorted(branches, key=lambda item: item["entries_today"], reverse=reverse)
    if sort_by == "exits":
        return sorted(branches, key=lambda item: item["exits_today"], reverse=reverse)
    return sorted(branches, key=lambda item: item["online_ratio"], reverse=reverse)


def _list_heatmap_slots(db: Session, camera_ids: list[str], limit: int = 72) -> list[dict[str, Any]]:
    if not camera_ids:
        return []

    rows = (
        db.query(
            Heatmap.date,
            Heatmap.hour,
            func.count(Heatmap.id).label("camera_count"),
        )
        .filter(Heatmap.camera_id.in_(camera_ids))
        .group_by(Heatmap.date, Heatmap.hour)
        .order_by(Heatmap.date.desc(), Heatmap.hour.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "date": row.date.date().isoformat() if isinstance(row.date, datetime) else str(row.date),
            "hour": int(row.hour),
            "camera_count": int(row.camera_count or 0),
        }
        for row in rows
    ]


def _pick_heatmap_slot(
    available_slots: list[dict[str, Any]],
    target_date: date | None,
    target_hour: int | None,
) -> tuple[date | None, int | None]:
    if not available_slots:
        return None, None

    if target_date is not None and target_hour is not None:
        return target_date, int(target_hour)

    if target_date is not None:
        target_date_iso = target_date.isoformat()
        same_day = [slot for slot in available_slots if slot["date"] == target_date_iso]
        if same_day:
            picked = max(same_day, key=lambda item: int(item["hour"]))
            return target_date, int(picked["hour"])

    latest = available_slots[0]
    return date.fromisoformat(latest["date"]), int(latest["hour"])


@router.get("/overview")
async def get_overview(
    hours: int = Query(default=24, ge=1, le=168),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    top_branches: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """
    Resumen global para página principal del dashboard.
    """
    cameras = db.query(CameraStatus).all()
    camera_ids = [camera.camera_id for camera in cameras]
    since, until, resolved_start_date, resolved_end_date = _resolve_time_window(
        hours=hours,
        start_date=start_date,
        end_date=end_date,
    )

    latest_counts = _latest_counts_by_camera(db, camera_ids)
    crossings_by_camera = _crossings_in_range_by_camera(db, camera_ids, since=since, until=until)
    branch_cards = _build_branch_cards(cameras, latest_counts, crossings_by_camera)
    flow_series = _hourly_flow(db, camera_ids, since=since, until=until)
    alerts = _build_alerts(cameras, latest_counts)

    entries_today = sum(item.get("entry", 0) for item in crossings_by_camera.values())
    exits_today = sum(item.get("exit", 0) for item in crossings_by_camera.values())
    online_cameras = sum(1 for camera in cameras if camera.is_connected)
    total_cameras = len(cameras)

    return {
        "timestamp": _utcnow().isoformat(),
        "window_hours": max(1, int((until - since).total_seconds() // 3600)),
        "start_date": resolved_start_date.isoformat(),
        "end_date": resolved_end_date.isoformat(),
        "total_branches": len(branch_cards),
        "total_cameras": total_cameras,
        "online_cameras": online_cameras,
        "offline_cameras": max(0, total_cameras - online_cameras),
        "camera_health": {
            "online": online_cameras,
            "offline": max(0, total_cameras - online_cameras),
            "high_error": sum(1 for camera in cameras if (camera.error_count or 0) >= 10),
            "total": total_cameras,
        },
        "current_occupancy": sum(latest_counts.values()),
        "entries_today": entries_today,
        "exits_today": exits_today,
        "net_today": entries_today - exits_today,
        "flow_series": flow_series,
        "top_branches": sorted(
            branch_cards,
            key=lambda item: (item["entries_today"], item["current_occupancy"]),
            reverse=True,
        )[:top_branches],
        "alerts": alerts[:10],
    }


@router.get("/branches")
async def list_branch_cards(
    q: str | None = Query(default=None, min_length=1),
    sort_by: Literal["name", "occupancy", "entries", "exits", "online_ratio"] = Query(default="name"),
    order: Literal["asc", "desc"] = Query(default="asc"),
    db: Session = Depends(get_db),
):
    """
    Tarjetas resumen por sucursal para vista global.
    """
    cameras = db.query(CameraStatus).all()
    camera_ids = [camera.camera_id for camera in cameras]
    latest_counts = _latest_counts_by_camera(db, camera_ids)
    today_start = datetime.combine(date.today(), datetime.min.time())
    tomorrow_start = today_start + timedelta(days=1)
    crossings_by_camera = _crossings_in_range_by_camera(
        db,
        camera_ids,
        since=today_start,
        until=tomorrow_start,
    )
    branches = _build_branch_cards(cameras, latest_counts, crossings_by_camera)

    if q:
        query = q.lower().strip()
        branches = [
            branch
            for branch in branches
            if query in branch["branch_name"].lower() or query in branch["branch_id"].lower()
        ]

    branches = _sort_branches(branches, sort_by=sort_by, order=order)
    entries_today = sum(branch["entries_today"] for branch in branches)
    exits_today = sum(branch["exits_today"] for branch in branches)

    return {
        "timestamp": _utcnow().isoformat(),
        "filters": {"q": q, "sort_by": sort_by, "order": order},
        "summary": {
            "entries_today": entries_today,
            "exits_today": exits_today,
            "net_today": entries_today - exits_today,
            "branch_count": len(branches),
        },
        "branches": branches,
    }


@router.get("/branches/{branch_id}")
async def get_branch_detail(
    branch_id: str,
    hours: int = Query(default=24, ge=1, le=168),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    db: Session = Depends(get_db)
):
    """
    Detalle de sucursal para dashboard.
    """
    cameras = db.query(CameraStatus).all()
    branch_cameras = [camera for camera in cameras if _camera_branch_info(camera)[0] == branch_id]
    camera_ids = [camera.camera_id for camera in branch_cameras]
    since, until, resolved_start_date, resolved_end_date = _resolve_time_window(
        hours=hours,
        start_date=start_date,
        end_date=end_date,
    )

    latest_counts = _latest_counts_by_camera(db, camera_ids)
    crossings_by_camera, crossings_gender_by_camera = _crossings_in_range_with_gender_by_camera(
        db,
        camera_ids,
        since=since,
        until=until,
    )
    flow_series = _hourly_flow(db, camera_ids, since=since, until=until)

    cameras_payload = []
    for camera in branch_cameras:
        cameras_payload.append(
            {
                "camera_id": camera.camera_id,
                "camera_name": camera.camera_name,
                "is_connected": camera.is_connected,
                "fps": camera.fps,
                "error_count": camera.error_count,
                "last_frame_at": camera.last_frame_at,
                "current_count": latest_counts.get(camera.camera_id, 0),
                "entry_today": crossings_by_camera.get(camera.camera_id, {}).get("entry", 0),
                "exit_today": crossings_by_camera.get(camera.camera_id, {}).get("exit", 0),
                "entry_by_gender": crossings_gender_by_camera.get(camera.camera_id, {}).get("entry", _empty_gender_bucket()),
                "exit_by_gender": crossings_gender_by_camera.get(camera.camera_id, {}).get("exit", _empty_gender_bucket()),
            }
        )

    entries_today = sum(item.get("entry", 0) for item in crossings_by_camera.values())
    exits_today = sum(item.get("exit", 0) for item in crossings_by_camera.values())
    online_cameras = sum(1 for camera in branch_cameras if camera.is_connected)
    branch_alerts = _build_alerts(branch_cameras, latest_counts)
    occupancy_peak = max((item["occupancy_end"] for item in flow_series), default=0)
    entries_by_gender = _sum_gender_counts(crossings_gender_by_camera, "entry")
    exits_by_gender = _sum_gender_counts(crossings_gender_by_camera, "exit")

    return {
        "timestamp": _utcnow().isoformat(),
        "window_hours": max(1, int((until - since).total_seconds() // 3600)),
        "start_date": resolved_start_date.isoformat(),
        "end_date": resolved_end_date.isoformat(),
        "branch_id": branch_id,
        "branch_name": _camera_branch_info(branch_cameras[0])[1] if branch_cameras else branch_id,
        "entries_today": entries_today,
        "exits_today": exits_today,
        "entries_by_gender": entries_by_gender,
        "exits_by_gender": exits_by_gender,
        "net_today": entries_today - exits_today,
        "current_occupancy": sum(latest_counts.values()),
        "occupancy_peak": occupancy_peak,
        "online_cameras": online_cameras,
        "total_cameras": len(branch_cameras),
        "online_ratio": round(online_cameras / max(1, len(branch_cameras)), 3),
        "cameras": cameras_payload,
        "alerts": branch_alerts[:10],
        "hourly_flow": flow_series,
    }


@router.get("/branches/{branch_id}/heatmaps")
async def get_branch_heatmaps(
    branch_id: str,
    target_date: date | None = Query(default=None),
    hour: int | None = Query(default=None, ge=0, le=23),
    db: Session = Depends(get_db),
):
    """
    Heatmaps horarios por sucursal (uno por cámara para el slot seleccionado).
    """
    cameras = db.query(CameraStatus).all()
    branch_cameras = [camera for camera in cameras if _camera_branch_info(camera)[0] == branch_id]
    camera_ids = [camera.camera_id for camera in branch_cameras]
    branch_name = _camera_branch_info(branch_cameras[0])[1] if branch_cameras else branch_id

    available_slots = _list_heatmap_slots(db, camera_ids=camera_ids, limit=96)
    selected_date, selected_hour = _pick_heatmap_slot(
        available_slots=available_slots,
        target_date=target_date,
        target_hour=hour,
    )

    selected_rows_by_camera: dict[str, Heatmap] = {}
    if selected_date is not None and selected_hour is not None and camera_ids:
        day_start = datetime.combine(selected_date, dt_time.min)
        rows = (
            db.query(Heatmap)
            .filter(
                Heatmap.camera_id.in_(camera_ids),
                Heatmap.date == day_start,
                Heatmap.hour == selected_hour,
            )
            .all()
        )
        selected_rows_by_camera = {row.camera_id: row for row in rows}

    cameras_payload = []
    for camera in branch_cameras:
        row = selected_rows_by_camera.get(camera.camera_id)
        heatmap_data = row.heatmap_data if row and isinstance(row.heatmap_data, dict) else None
        cameras_payload.append(
            {
                "camera_id": camera.camera_id,
                "camera_name": camera.camera_name,
                "is_connected": bool(camera.is_connected),
                "last_frame_at": camera.last_frame_at,
                "heatmap": heatmap_data,
            }
        )

    populated = sum(1 for camera in cameras_payload if camera["heatmap"] is not None)
    return {
        "timestamp": _utcnow().isoformat(),
        "branch_id": branch_id,
        "branch_name": branch_name,
        "selected_slot": (
            {"date": selected_date.isoformat(), "hour": selected_hour}
            if selected_date is not None and selected_hour is not None
            else None
        ),
        "available_slots": available_slots,
        "total_cameras": len(branch_cameras),
        "cameras_with_heatmap": populated,
        "cameras": cameras_payload,
    }

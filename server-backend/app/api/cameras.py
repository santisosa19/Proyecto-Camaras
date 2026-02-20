"""
API Router - Cámaras
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime

from app.database.connection import get_db
from app.models.database import CameraStatus

router = APIRouter()


class CameraInfo(BaseModel):
    """Schema para información de cámara"""
    camera_id: str
    camera_name: str
    rtsp_url: str
    is_active: bool
    is_connected: bool
    fps: float
    total_frames: int
    error_count: int
    last_frame_at: datetime | None
    
    class Config:
        from_attributes = True


class CameraCreate(BaseModel):
    """Schema para crear cámara"""
    camera_id: str
    camera_name: str
    rtsp_url: str
    fps: int = 15


@router.get("/", response_model=List[CameraInfo])
async def list_cameras(db: Session = Depends(get_db)):
    """
    Listar todas las cámaras registradas
    """
    cameras = db.query(CameraStatus).all()
    return cameras


@router.get("/{camera_id}", response_model=CameraInfo)
async def get_camera(camera_id: str, db: Session = Depends(get_db)):
    """
    Obtener información de una cámara específica
    """
    camera = db.query(CameraStatus).filter(
        CameraStatus.camera_id == camera_id
    ).first()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    return camera


@router.post("/", response_model=CameraInfo)
async def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """
    Registrar una nueva cámara
    """
    # Verificar si ya existe
    existing = db.query(CameraStatus).filter(
        CameraStatus.camera_id == camera.camera_id
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Cámara {camera.camera_id} ya existe"
        )
    
    # Crear nueva
    new_camera = CameraStatus(
        camera_id=camera.camera_id,
        camera_name=camera.camera_name,
        rtsp_url=camera.rtsp_url,
        is_active=True,
        is_connected=False
    )
    
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)
    
    return new_camera


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    """
    Eliminar una cámara
    """
    camera = db.query(CameraStatus).filter(
        CameraStatus.camera_id == camera_id
    ).first()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    db.delete(camera)
    db.commit()
    
    return {"message": f"Cámara {camera_id} eliminada exitosamente"}


@router.post("/{camera_id}/activate")
async def activate_camera(camera_id: str, db: Session = Depends(get_db)):
    """
    Activar una cámara
    """
    camera = db.query(CameraStatus).filter(
        CameraStatus.camera_id == camera_id
    ).first()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    camera.is_active = True
    db.commit()
    
    return {"message": f"Cámara {camera_id} activada"}


@router.post("/{camera_id}/deactivate")
async def deactivate_camera(camera_id: str, db: Session = Depends(get_db)):
    """
    Desactivar una cámara
    """
    camera = db.query(CameraStatus).filter(
        CameraStatus.camera_id == camera_id
    ).first()
    
    if not camera:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    camera.is_active = False
    db.commit()
    
    return {"message": f"Cámara {camera_id} desactivada"}

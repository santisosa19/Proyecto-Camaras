"""
API Router - Procesamiento
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

router = APIRouter()


# Diccionario global para tracking del estado de procesamiento
# En producción, esto estaría en Redis o similar
processing_state: Dict[str, dict] = {}


class ProcessorStatus(BaseModel):
    """Estado del procesador"""
    camera_id: str
    is_running: bool
    uptime_seconds: float
    frames_processed: int
    current_detections: int
    errors: int


@router.get("/status")
async def get_all_status():
    """
    Obtener estado de todos los procesadores
    """
    return {
        "processors": processing_state,
        "total_active": sum(1 for p in processing_state.values() if p.get('is_running', False))
    }


@router.get("/status/{camera_id}")
async def get_processor_status(camera_id: str):
    """
    Obtener estado de un procesador específico
    """
    if camera_id not in processing_state:
        raise HTTPException(
            status_code=404,
            detail=f"Procesador para cámara {camera_id} no encontrado"
        )
    
    return processing_state[camera_id]


@router.post("/start/{camera_id}")
async def start_processor(camera_id: str):
    """
    Iniciar procesador de una cámara
    
    TODO: Implementar lógica real de inicio
    """
    if camera_id in processing_state and processing_state[camera_id].get('is_running'):
        raise HTTPException(
            status_code=400,
            detail=f"Procesador para {camera_id} ya está corriendo"
        )
    
    # En producción, aquí se iniciaría el procesador real
    processing_state[camera_id] = {
        'is_running': True,
        'started_at': 'datetime.now()',
        'frames_processed': 0,
        'errors': 0
    }
    
    return {
        "message": f"Procesador para {camera_id} iniciado",
        "status": processing_state[camera_id]
    }


@router.post("/stop/{camera_id}")
async def stop_processor(camera_id: str):
    """
    Detener procesador de una cámara
    """
    if camera_id not in processing_state:
        raise HTTPException(
            status_code=404,
            detail=f"Procesador para {camera_id} no encontrado"
        )
    
    # En producción, aquí se detendría el procesador real
    if processing_state[camera_id].get('is_running'):
        processing_state[camera_id]['is_running'] = False
    
    return {
        "message": f"Procesador para {camera_id} detenido",
        "status": processing_state[camera_id]
    }


@router.post("/restart/{camera_id}")
async def restart_processor(camera_id: str):
    """
    Reiniciar procesador de una cámara
    """
    # Detener
    if camera_id in processing_state:
        processing_state[camera_id]['is_running'] = False
    
    # Iniciar
    processing_state[camera_id] = {
        'is_running': True,
        'started_at': 'datetime.now()',
        'frames_processed': 0,
        'errors': 0
    }
    
    return {
        "message": f"Procesador para {camera_id} reiniciado",
        "status": processing_state[camera_id]
    }

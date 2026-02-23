"""
Aplicación principal FastAPI - Traffic Analysis System
"""
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from app.config import settings
from app.database.connection import DatabaseManager, get_db_context, init_db

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def _hourly_aggregation_worker():
    """
    Worker periódico que agrega hourly_metrics y limpia detections de horas cerradas.
    """
    while True:
        try:
            with get_db_context() as db:
                result = DatabaseManager.aggregate_completed_hourly_metrics(db)
                if result["processed_hours"] > 0:
                    logger.info(
                        "✓ Agregación horaria: horas=%s, detections borradas=%s",
                        result["processed_hours"],
                        result["deleted_detections"],
                    )
        except Exception as exc:
            logger.error(f"Error en agregación horaria automática: {exc}", exc_info=True)

        await asyncio.sleep(60)


# Lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle de la aplicación"""
    # Startup
    logger.info("🚀 Iniciando Traffic Analysis System...")
    
    try:
        # Inicializar base de datos
        init_db()
        logger.info("✓ Base de datos inicializada")

        # Iniciar agregador horario automático
        app.state.hourly_aggregation_task = asyncio.create_task(_hourly_aggregation_worker())
        logger.info("✓ Worker de agregación horaria iniciado")
        
        # Aquí se pueden agregar otros servicios de inicio
        # Por ejemplo: iniciar procesadores de cámara
        
        logger.info("✓ Sistema iniciado exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Apagando Traffic Analysis System...")

    task = getattr(app.state, "hourly_aggregation_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Liberar recursos aquí
    logger.info("✓ Sistema apagado")


# Crear aplicación
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Sistema de análisis de tráfico con visión computacional",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# ROUTES PRINCIPALES
# ============================================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.ENVIRONMENT
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/info")
async def get_info():
    """Información del sistema"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "features": {
            "video_capture": True,
            "yolo_detection": True,
            "line_counting": True,
            "zone_counting": True,
            "heatmaps": True,
            "cegid_integration": bool(settings.CEGID_DB_URL)
        }
    }


# ============================================
# IMPORTAR ROUTERS
# ============================================

from app.api import cameras, dashboard, ingest, metrics, processing

# Registrar routers
app.include_router(
    cameras.router,
    prefix="/api/v1/cameras",
    tags=["cameras"]
)

app.include_router(
    metrics.router,
    prefix="/api/v1/metrics",
    tags=["metrics"]
)

app.include_router(
    processing.router,
    prefix="/api/v1/processing",
    tags=["processing"]
)

app.include_router(
    ingest.router,
    prefix="/api/v1/ingest",
    tags=["ingest"]
)

app.include_router(
    dashboard.router,
    prefix="/api/v1/dashboard",
    tags=["dashboard"]
)


# ============================================
# WEBSOCKET
# ============================================

class ConnectionManager:
    """Manager para conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket conectado. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket desconectado. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Enviar mensaje a todos los clientes conectados"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error enviando mensaje WebSocket: {e}")


manager = ConnectionManager()


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket para métricas en tiempo real
    
    Envía updates periódicos con métricas de todas las cámaras
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # El cliente puede enviar comandos
            data = await websocket.receive_json()
            
            # Procesar comando (por ejemplo, cambiar cámara)
            if data.get("command") == "subscribe":
                camera_id = data.get("camera_id")
                logger.info(f"Cliente suscrito a cámara {camera_id}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Cliente WebSocket desconectado")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        manager.disconnect(websocket)


# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global de excepciones"""
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================
# STARTUP TASK EXAMPLE
# ============================================

@app.on_event("startup")
async def startup_event():
    """Tareas de inicio adicionales"""
    logger.info("Ejecutando tareas de inicio...")
    
    # Aquí se pueden iniciar procesadores de cámara en background
    # import asyncio
    # asyncio.create_task(start_camera_processor())


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=1 if settings.API_RELOAD else settings.API_WORKERS
    )

"""
Aplicación principal FastAPI - Traffic Analysis System
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

import redis
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.database.connection import DatabaseManager, engine, get_db_context, init_db

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _check_database() -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def _check_redis() -> None:
    client = redis.Redis.from_url(
        settings.REDIS_URL,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    try:
        client.ping()
    finally:
        client.close()


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

        if not settings.INGEST_API_KEY.strip():
            logger.warning("INGEST_API_KEY no está configurada; la ingesta quedará abierta.")
        if not settings.ADMIN_API_KEY.strip():
            logger.warning("ADMIN_API_KEY no está configurada; endpoints administrativos sin protección.")

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


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "Request failed req_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    logger.info(
        "Request req_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


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
    """
    Health check con validación de dependencias.
    Devuelve 503 si algún servicio crítico no está disponible.
    """
    checks = {"database": "ok", "redis": "ok"}

    try:
        _check_database()
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    try:
        _check_redis()
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    status = "healthy" if all(value == "ok" for value in checks.values()) else "unhealthy"
    payload = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "services": checks,
        "environment": settings.ENVIRONMENT,
    }

    if status == "healthy":
        return payload

    return JSONResponse(status_code=503, content=payload)


@app.get("/health/live")
async def liveness_check():
    """
    Liveness probe: verifica que el proceso responde.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/ready")
async def readiness_check():
    """
    Readiness probe: alias de /health para integraciones de orquestación.
    """
    return await health_check()


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

from app.api import auth, cameras, dashboard, ingest, metrics, processing

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

app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["auth"]
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

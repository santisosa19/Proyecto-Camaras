"""
Configuración central de la aplicación
"""
import json
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import List

# Determinar la ruta al .env
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # App
    APP_NAME: str = "Traffic Analysis System"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 0
    
    # Redis
    REDIS_URL: str
    REDIS_MAX_CONNECTIONS: int = 50
    
    # Cegid
    CEGID_DB_URL: str = ""
    CEGID_SYNC_INTERVAL: int = 3600
    
    # Security
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 1440
    ADMIN_API_KEY: str = ""
    AUTH_ACCESS_TOKEN_EXPIRE_MINUTES: int = 720
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # YOLO Detection
    YOLO_MODEL: str = "yolov8n.pt"
    DETECTION_CONFIDENCE: float = 0.5
    TRACKING_MAX_AGE: int = 30
    TRACKING_MIN_HITS: int = 3
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/traffic_system.log"
    
    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # Storage
    HEATMAP_STORAGE_PATH: str = "/data/heatmaps"
    BACKUP_PATH: str = "/data/backups"
    RETENTION_DAYS: int = 90
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Ingest API (sucursales -> central)
    INGEST_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        """
        Soporta CORS_ORIGINS como lista o string separado por comas.
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return json.loads(stripped)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str):
        level = value.upper().strip()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"LOG_LEVEL inválido: {value}. Permitidos: {sorted(allowed)}")
        return level

    @model_validator(mode="after")
    def validate_runtime_safety(self):
        env = self.ENVIRONMENT.lower().strip()
        if env == "production":
            if self.DEBUG:
                raise ValueError("DEBUG debe ser false en producción")
            if self.API_RELOAD:
                raise ValueError("API_RELOAD debe ser false en producción")

            weak_jwt_values = {"", "change-this-secret", "changeme", "secret"}
            if self.JWT_SECRET_KEY.strip().lower() in weak_jwt_values:
                raise ValueError("JWT_SECRET_KEY inválido para producción")

            if not self.INGEST_API_KEY.strip():
                raise ValueError("INGEST_API_KEY es obligatorio en producción")

            if not self.ADMIN_API_KEY.strip():
                raise ValueError("ADMIN_API_KEY es obligatorio en producción")

            if not self.CORS_ORIGINS:
                raise ValueError("CORS_ORIGINS no puede estar vacío en producción")
        return self


@lru_cache()
def get_settings() -> Settings:
    """Obtener configuración singleton"""
    return Settings()


# Instancia global
settings = get_settings()

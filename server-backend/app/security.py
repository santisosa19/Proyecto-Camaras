"""
Dependencias de seguridad para endpoints administrativos.
"""
from fastapi import Header, HTTPException, status

from app.config import settings


def require_admin_api_key(
    x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key")
):
    """
    Si ADMIN_API_KEY está configurada, exige header X-Admin-API-Key válido.
    En desarrollo puede quedar vacío para facilitar pruebas locales.
    """
    required_key = settings.ADMIN_API_KEY.strip()
    if not required_key:
        if settings.ENVIRONMENT.lower().strip() == "production":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ADMIN_API_KEY no está configurada en el servidor",
            )
        return

    if x_admin_api_key != required_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key administrativa inválida",
        )

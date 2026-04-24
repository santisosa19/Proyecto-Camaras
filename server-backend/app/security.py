"""
Dependencias de seguridad para endpoints administrativos y login JWT.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from hmac import compare_digest

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import get_db
from app.models.database import AppUser


PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 390000
VALID_ROLES = {"superadmin", "manager", "branch_manager"}


class AuthenticatedUser(BaseModel):
    user_id: int
    username: str
    role: str
    branch_id: str | None = None


_bearer_scheme = HTTPBearer(auto_error=False)


def _auth_error(detail: str = "Token inválido") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def normalize_role(role: str) -> str:
    clean = (role or "").strip().lower()
    return clean if clean in VALID_ROLES else "branch_manager"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations_raw, salt_hex, digest_hex = stored_hash.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(iterations_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return compare_digest(candidate, expected)


def authenticate_user(db: Session, username: str, password: str) -> AuthenticatedUser | None:
    username_clean = username.strip()
    if not username_clean:
        return None

    user = db.query(AppUser).filter(AppUser.username == username_clean).first()
    if user is None or not user.is_active:
        return None
    if not verify_password(password, user.password_hash):
        return None

    return AuthenticatedUser(
        user_id=int(user.id),
        username=user.username,
        role=normalize_role(user.role),
        branch_id=user.branch_id,
    )


def create_access_token(user: AuthenticatedUser) -> str:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=max(1, int(settings.AUTH_ACCESS_TOKEN_EXPIRE_MINUTES)))
    payload = {
        "sub": str(user.user_id),
        "username": user.username,
        "role": user.role,
        "branch_id": user.branch_id,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _auth_error("Token requerido")

    token = credentials.credentials.strip()
    if not token:
        raise _auth_error("Token vacío")

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise _auth_error("Token expirado") from exc
    except jwt.InvalidTokenError as exc:
        raise _auth_error("Token inválido") from exc

    user_id_raw = str(payload.get("sub", "")).strip()
    if not user_id_raw.isdigit():
        raise _auth_error("Token inválido")
    user_id = int(user_id_raw)

    user = db.query(AppUser).filter(AppUser.id == user_id).first()
    if user is None or not user.is_active:
        raise _auth_error("Usuario inválido o inactivo")

    return AuthenticatedUser(
        user_id=int(user.id),
        username=user.username,
        role=normalize_role(user.role),
        branch_id=user.branch_id,
    )


def require_superadmin(current_user: AuthenticatedUser = Depends(require_authenticated_user)) -> AuthenticatedUser:
    if current_user.role != "superadmin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Se requiere rol superadmin",
        )
    return current_user


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

"""
API Router - Autenticación
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database.connection import get_db
from app.models.database import AppUser
from app.security import (
    AuthenticatedUser,
    authenticate_user,
    create_access_token,
    hash_password,
    require_admin_api_key,
    require_authenticated_user,
    require_superadmin,
    normalize_role,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: AuthenticatedUser


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=200)
    role: str = Field(default="branch_manager", min_length=1, max_length=30)
    branch_id: str | None = Field(default=None, max_length=50)


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    branch_id: str | None = None
    is_active: bool


class BootstrapAdminRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=200)


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    if db.query(AppUser).count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No hay usuarios en base de datos. Ejecutá bootstrap-admin primero.",
        )

    user = authenticate_user(db=db, username=payload.username, password=payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña inválidos",
        )

    token = create_access_token(user)
    return LoginResponse(
        access_token=token,
        expires_in_minutes=max(1, int(settings.AUTH_ACCESS_TOKEN_EXPIRE_MINUTES)),
        user=user,
    )


@router.get("/me", response_model=AuthenticatedUser)
async def get_me(current_user: AuthenticatedUser = Depends(require_authenticated_user)):
    return current_user


@router.post("/users", response_model=UserResponse)
async def create_user(
    payload: CreateUserRequest,
    _: AuthenticatedUser = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    username = payload.username.strip()
    existing = db.query(AppUser).filter(AppUser.username == username).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"El usuario '{username}' ya existe",
        )

    role = normalize_role(payload.role)
    user = AppUser(
        username=username,
        password_hash=hash_password(payload.password),
        role=role,
        branch_id=(payload.branch_id.strip() if payload.branch_id else None),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(
        id=int(user.id),
        username=user.username,
        role=role,
        branch_id=user.branch_id,
        is_active=bool(user.is_active),
    )


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    _: AuthenticatedUser = Depends(require_superadmin),
    db: Session = Depends(get_db),
):
    users = db.query(AppUser).order_by(AppUser.username.asc()).all()
    return [
        UserResponse(
            id=int(user.id),
            username=user.username,
            role=normalize_role(user.role),
            branch_id=user.branch_id,
            is_active=bool(user.is_active),
        )
        for user in users
    ]


@router.post("/bootstrap-admin", response_model=UserResponse)
async def bootstrap_admin(
    payload: BootstrapAdminRequest,
    _admin_key: None = Depends(require_admin_api_key),
    db: Session = Depends(get_db),
):
    total_users = db.query(AppUser).count()
    if total_users > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bootstrap no disponible: ya existen usuarios en la base de datos",
        )

    user = AppUser(
        username=payload.username.strip(),
        password_hash=hash_password(payload.password),
        role="superadmin",
        branch_id=None,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=int(user.id),
        username=user.username,
        role=user.role,
        branch_id=user.branch_id,
        is_active=bool(user.is_active),
    )

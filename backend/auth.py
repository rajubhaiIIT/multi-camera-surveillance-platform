"""Authentication, JWT token lifecycle, and Role-Based Access Control (RBAC)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .config import settings
from .schemas import UserResponse, UserRole

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Built-in demo accounts for instant platform access
_DEFAULT_USERS: dict[str, dict] = {
    "admin": {
        "username": "admin",
        "password_hash": bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
        "role": UserRole.ADMIN,
        "full_name": "System Administrator",
        "email": "admin@surveillance.local",
        "is_active": True,
        "created_at": time.time(),
    },
    "operator": {
        "username": "operator",
        "password_hash": bcrypt.hashpw("operator123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
        "role": UserRole.OPERATOR,
        "full_name": "Security Operations Center",
        "email": "soc@surveillance.local",
        "is_active": True,
        "created_at": time.time(),
    },
    "viewer": {
        "username": "viewer",
        "password_hash": bcrypt.hashpw("viewer123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
        "role": UserRole.VIEWER,
        "full_name": "Auditor / Read-Only",
        "email": "viewer@surveillance.local",
        "is_active": True,
        "created_at": time.time(),
    },
}

# In-memory user registry
USERS_DB: dict[str, dict] = dict(_DEFAULT_USERS)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def authenticate_user(username: str, password: str) -> dict | None:
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserResponse:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = USERS_DB.get(username)
    if user is None or not user.get("is_active", True):
        raise credentials_exception

    return UserResponse(
        username=user["username"],
        role=user["role"],
        full_name=user.get("full_name"),
        email=user.get("email"),
        is_active=user.get("is_active", True),
        created_at=user.get("created_at", time.time()),
    )


def require_role(allowed_roles: list[UserRole]):
    """Role-based authorization dependency guard."""
    async def role_checker(current_user: Annotated[UserResponse, Depends(get_current_user)]) -> UserResponse:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {[r.value for r in allowed_roles]}, user has: {current_user.role.value}"
            )
        return current_user
    return role_checker

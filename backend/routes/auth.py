"""Authentication API routes."""
from __future__ import annotations

import time
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ..auth import (
    USERS_DB,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
    require_role,
)
from ..config import settings
from ..schemas import LoginRequest, TokenResponse, UserCreate, UserResponse, UserRole

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expires = timedelta(minutes=settings.access_token_expire_minutes)
    token = create_access_token(
        data={"sub": user["username"], "role": user["role"].value},
        expires_delta=expires,
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        username=user["username"],
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/login", response_model=TokenResponse)
async def login_json(payload: LoginRequest):
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    expires = timedelta(minutes=settings.access_token_expire_minutes)
    token = create_access_token(
        data={"sub": user["username"], "role": user["role"].value},
        expires_delta=expires,
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        username=user["username"],
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: Annotated[UserResponse, Depends(get_current_user)]):
    return current_user


@router.get("/users", response_model=list[UserResponse], dependencies=[Depends(require_role([UserRole.ADMIN]))])
async def list_users():
    return [
        UserResponse(
            username=u["username"],
            role=u["role"],
            full_name=u.get("full_name"),
            email=u.get("email"),
            is_active=u.get("is_active", True),
            created_at=u.get("created_at", time.time()),
        )
        for u in USERS_DB.values()
    ]


@router.post("/users", response_model=UserResponse, dependencies=[Depends(require_role([UserRole.ADMIN]))])
async def create_user(user_in: UserCreate):
    if user_in.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    now = time.time()
    USERS_DB[user_in.username] = {
        "username": user_in.username,
        "password_hash": get_password_hash(user_in.password),
        "role": user_in.role,
        "full_name": user_in.full_name,
        "email": user_in.email,
        "is_active": True,
        "created_at": now,
    }
    return UserResponse(
        username=user_in.username,
        role=user_in.role,
        full_name=user_in.full_name,
        email=user_in.email,
        is_active=True,
        created_at=now,
    )

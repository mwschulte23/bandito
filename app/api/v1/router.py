from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth, bandit
)


api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(bandit.router, prefix="/bandit", tags=["bandit"])
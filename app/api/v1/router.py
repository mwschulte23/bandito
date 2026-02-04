from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth, bandit, bandit_actions, bandit_analysis
)


api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(bandit.router, prefix="/bandit", tags=["bandit"])
api_router.include_router(bandit_actions.router, prefix="/bandit", tags=["bandit_actions"])
api_router.include_router(bandit_analysis.router, prefix="/bandit", tags=["bandit_analysis"])
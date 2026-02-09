# import stripe
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.db.session import get_session, init_db

from app.models.user import User
from app.models.bandit import Bandit, BanditMode, BanditState
from app.schemas.bandit import BanditStateInternal



app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    root_url=''
)
app.include_router(api_router, prefix=settings.API_V1_STR)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check(session: AsyncSession = Depends(get_session)):
   try:
       await session.execute(select(1))
       return {"status": "healthy"}
   except Exception:
       return {"status": "unhealthy"}

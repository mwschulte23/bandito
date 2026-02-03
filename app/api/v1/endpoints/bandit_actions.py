import io
import numpy as np
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.core.deps import verify_user
from app.core.deps import get_current_user
from app.models.user import User
from app.models.bandit import Bandit, BanditArm, BanditState, BanditEvent, EventSegment
# from app.schemas.bandit import (
#     BanditCreate, BanditRead, BanditUpdate, BanditReadWithArms,
#     BanditArmCreate, BanditArmRead, BanditArmUpdate,
#     BanditStateResponse,
#     BanditEventUpdate, BanditEventCreateWithSegments, BanditEventResponse,
#     EventSegmentRead,
# )


router = APIRouter(dependencies=[Depends(verify_user)])


@router.post()

# pull arm
# reward
"""
Data helper functions for bandit operations.

All functions accept an AsyncSession parameter for proper transaction management.
"""

import numpy as np
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.schemas.bandit import BanditRead, BanditReadArmsState, BanditArmRead, BanditStateInternal
from app.models.bandit import Bandit, BanditArm, BanditState, BanditEvent, EventSegment


async def get_full_bandit(
    session: AsyncSession, bandit_id: int, user_id: int
) -> BanditReadArmsState:
    """
    Fetch bandit with arms and state.

    Assumes user/bandit relationship has been validated already.
    """
    result = await session.execute(
        select(Bandit)
        .where(Bandit.id == bandit_id, Bandit.user_id == user_id)
        .options(selectinload(Bandit.arms), selectinload(Bandit.state))
    )
    bandit = result.scalar_one_or_none()
    if not bandit:
        raise HTTPException(status_code=404, detail=f"Bandit {bandit_id} not found")
    return BanditReadArmsState(
        **BanditRead.model_validate(bandit).model_dump(),
        arms=[BanditArmRead.model_validate(arm) for arm in bandit.arms],
        state=BanditStateInternal.from_db(bandit.state)
    )


async def add_event(
    session: AsyncSession,
    bandit_id: int,
    arm_id: int,
    pull_score: float,
    context: dict,
    user_query: str | dict
) -> int:
    """Create a new bandit event and return its ID."""
    event = BanditEvent(
        bandit_id=bandit_id,
        arm_id=arm_id,
        model_score=pull_score,
        context=context,
        user_query=user_query
    )
    session.add(event)
    await session.flush()
    return event.id


async def get_event(session: AsyncSession, event_id: int) -> BanditEvent:
    """Fetch an event by ID."""
    result = await session.execute(
        select(BanditEvent).where(BanditEvent.id == event_id)
    )
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return event


async def update_event(session: AsyncSession, event: BanditEvent) -> None:
    """Mark event for update (already modified by caller)."""
    session.add(event)


async def update_state(
    session: AsyncSession, state_id: int, new_state: BanditStateInternal
) -> None:
    """Update bandit state with new values."""
    result = await session.execute(
        select(BanditState).where(BanditState.id == state_id)
    )
    state = result.scalar_one_or_none()
    if not state:
        raise HTTPException(status_code=404, detail="State not found")

    new_encoded_state = new_state.to_db()
    for key, value in new_encoded_state.model_dump(exclude_unset=True).items():
        setattr(state, key, value)

    session.add(state)

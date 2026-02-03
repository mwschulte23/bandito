import io
import asyncio
import numpy as np
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.core.deps import get_current_user
from app.schemas.bandit import BanditRead, BanditReadArmsState, BanditArmRead, BanditStateRead
from app.models.bandit import Bandit, BanditArm, BanditState, BanditEvent, EventSegment

'''
Approach: We have three distinct "things" happening
1. An arm selection
2. A multi-stage reward + update system: Immediate, human and outcome
3. Event management

These are intertwined with the "arm pull" kicking things off. The process.
1. Arm pull -> 
    * In: arm configs, context and state
    * Return: selected arm
    * Data: GET /{bandit_id} -> arms, state; POST /{bandit_id} -> event "frame"
        * Opt. - user can provide user context via segments
2. Immediate reward ->
    * In: state and event frame (primarily features)
    * Out: Nothing needed
    * Internally, we fill-in reward details for event - update and save state --- need to SAVE LLM output
        * Maybe save LLM outputs can be all sorts of things...but needed for human eval side

Beyond this process, we can run full updates on selected events / segments. Core idea is analysis and bandit "splitting" feature.
* We recalculate bandit state for trailing N events
* Or on set of events for analysis, etc

SO...our data_helper needs to 
1. GET -> bandit, arms, state
2. POST/PATCH -> events, state
'''

# approach -> 

# get a bandit
    # including state
# pick an arm (get arms, pick one)

async def get_full_bandit(bandit_id: int, user_id: int) -> int:
    '''
    ** assumes user / bandit relationship has been validated already **
    returns bandit with arms
    '''
    async for session in get_session():
        result = await session.execute(
            select(Bandit)
            .where(Bandit.id == bandit_id, Bandit.user_id == user_id)
            .options(selectinload(Bandit.arms), selectinload(Bandit.state))
        )
        bandit = result.scalar_one_or_none()
        if not bandit:
            raise Exception(f"Bandit {bandit_id} not found")
        return BanditReadArmsState(
            **BanditRead.model_validate(bandit).model_dump(),
            arms=[BanditArmRead.model_validate(arm) for arm in bandit.arms],
            state=BanditStateRead.from_db(bandit.state)
        )


async def add_event(
    bandit_id: int, arm_id: int, pull_score: float,
    context: dict, user_query: str | dict
) -> int:
    async for session in get_session():
        event = BanditEvent(
            bandit_id=bandit_id,
            arm_id=arm_id,
            model_score=pull_score,
            context=context,
            user_query=user_query
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        return event.id


async def get_event(event_id: int):
    async for session in get_session():
        result = await session.execute(
            select(BanditEvent)
            .where(BanditEvent.id == event_id)
        )
        event = result.scalar_one_or_none()
        if not event:
            raise Exception(f"Event {event_id} not found")
        return event


async def update_event(event_id: int, new_event: BanditStateRead):
    async for session in get_session():
        result = await session.execute(
            select(BanditEvent)
            .where(BanditEvent.id == event_id)
        )
        event = result.scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        for key, value in new_event.model_dump(exclude_unset=True).items():
            setattr(event, key, value)

        session.add(event)
        await session.commit()
        await session.refresh(event)


async def update_state(state_id: int, new_state: BanditStateRead):
    async for session in get_session():
        result = await session.execute(
            select(BanditState)
            .where(BanditState.id == state_id)
        )
        state = result.scalar_one_or_none()
        if not state:
            raise HTTPException(status_code=404, detail="State not found")
        new_encoded_state = new_state.to_db()
        for key, value in new_encoded_state.model_dump(exclude_unset=True).items():
            setattr(state, key, value)

        session.add(state)
        await session.commit()
        await session.refresh(state)



if __name__ == '__main__':
    resp = asyncio.run(get_full_bandit(2, 1))
    print(resp.state)
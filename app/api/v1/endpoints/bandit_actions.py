from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.core.deps import verify_user, get_current_user
from app.models.user import User
from app.models.bandit import Bandit, BanditArm, BanditEvent
from app.services.bandit.bandit_brain import pull_arm, update_on_reward
from app.services.bandit.helpers import get_bandit_for_user, calculate_budget_status
from app.schemas.bandit_actions import PullArmRequest, PullArmResponse, ImmediateRewardRequest, HumanRewardRequest
from app.schemas.bandit import BanditArmRead, BanditEventRead


router = APIRouter(dependencies=[Depends(verify_user)])


@router.post("/{bandit_id}/pull", response_model=PullArmResponse, tags=["bandit_actions"])
async def pull_arm_endpoint(
    bandit_id: int,
    request: PullArmRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Select an arm using Thompson Sampling.

    Returns the selected arm configuration for the client to call the LLM.
    The event_id should be used when submitting the reward.

    Includes budget_warning if >= 90% of budget consumed.
    """
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    # Budget warning check
    budget_status = await calculate_budget_status(session, bandit_id, bandit.budget)
    budget_warning = budget_status["budget_warning"]

    result = await pull_arm(
        session=session,
        bandit_id=bandit_id,
        user_id=current_user.id,
        user_query=request.user_query,
        gamma=request.gamma,
        beta=request.beta
    )

    # Commit the event created by pull_arm
    await session.commit()

    # Fetch full arm details
    arm_result = await session.execute(
        select(BanditArm).where(BanditArm.id == result['arm_id'])
    )
    arm = arm_result.scalar_one_or_none()
    if not arm:
        raise HTTPException(status_code=500, detail="Selected arm not found")

    return PullArmResponse(
        event_id=result['event_id'],
        arm=BanditArmRead.model_validate(arm),
        context=result['context'],
        budget_warning=budget_warning
    )


@router.post(
    "/{bandit_id}/events/{event_id}/reward/immediate",
    response_model=BanditEventRead,
    tags=["bandit_actions"]
)
async def immediate_reward_endpoint(
    bandit_id: int,
    event_id: int,
    request: ImmediateRewardRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Submit immediate reward after LLM response.

    Called right after the LLM call with the response, cost, and latency.
    Updates bandit state with the reward signal.
    """
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    event_result = await session.execute(
        select(BanditEvent).where(
            BanditEvent.id == event_id,
            BanditEvent.bandit_id == bandit_id
        )
    )
    event = event_result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    await update_on_reward(
        session=session,
        bandit_id=bandit_id,
        user_id=current_user.id,
        event_id=event_id,
        llm_output=request.llm_output,
        reward=request.score,
        cost=request.cost,
        latency=request.latency,
        is_human_reward=False
    )

    # Commit the updates
    await session.commit()

    event_result = await session.execute(
        select(BanditEvent).where(BanditEvent.id == event_id)
    )
    updated_event = event_result.scalar_one()

    return BanditEventRead.model_validate(updated_event)


@router.post(
    "/{bandit_id}/events/{event_id}/reward/human",
    response_model=BanditEventRead,
    tags=["bandit_actions"]
)
async def human_reward_endpoint(
    bandit_id: int,
    event_id: int,
    request: HumanRewardRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Submit human feedback for an event.

    Called when a human rates the LLM response (0 = bad, 1 = good).
    Updates bandit state with the human reward signal.
    """
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    event_result = await session.execute(
        select(BanditEvent).where(
            BanditEvent.id == event_id,
            BanditEvent.bandit_id == bandit_id
        )
    )
    event = event_result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    await update_on_reward(
        session=session,
        bandit_id=bandit_id,
        user_id=current_user.id,
        event_id=event_id,
        llm_output=None,
        reward=request.score,
        cost=None,
        latency=None,
        is_human_reward=True
    )

    # Commit the updates
    await session.commit()

    event_result = await session.execute(
        select(BanditEvent).where(BanditEvent.id == event_id)
    )
    updated_event = event_result.scalar_one()

    return BanditEventRead.model_validate(updated_event)

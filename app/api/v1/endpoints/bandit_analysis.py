from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.session import get_session
from app.core.deps import verify_user, get_current_user
from app.models.user import User
from app.models.bandit import Bandit, BanditEvent
from app.schemas.bandit import BanditArmRead, BanditStateInternal
from app.schemas.bandit_actions import LeaderboardResponse, StateAnalysisResponse, BudgetResponse
from app.services.bandit.bandit_analysis import get_event_leaderboard, get_state_analysis
from app.services.bandit.helpers import get_bandit_for_user, calculate_budget_status


router = APIRouter(dependencies=[Depends(verify_user)])


@router.get("/{bandit_id}/leaderboard", response_model=LeaderboardResponse, tags=["bandit_analysis"])
async def get_leaderboard(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get event-based leaderboard for a bandit.

    Shows per-arm stats: pull count, average rewards, cost, latency.
    Sorted by average immediate reward (descending).
    """
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    # Get total pull count
    total_result = await session.execute(
        select(func.count(BanditEvent.id)).where(BanditEvent.bandit_id == bandit_id)
    )
    total_pulls = total_result.scalar() or 0

    # Get per-arm leaderboard
    arms = await get_event_leaderboard(session, bandit_id)

    return LeaderboardResponse(
        bandit_id=bandit_id,
        total_pulls=total_pulls,
        arms=arms
    )


@router.get("/{bandit_id}/analysis", response_model=StateAnalysisResponse, tags=["bandit_analysis"])
async def get_analysis(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get state-based analysis for a bandit.

    Decomposes theta_hat into interpretable components:
    - Model baseline effects
    - Prompt effects
    - Time interaction weights (hour_sin, hour_cos, weekend)
    - Uncertainty estimates per feature
    """
    # Fetch bandit with arms and state
    result = await session.execute(
        select(Bandit)
        .where(Bandit.id == bandit_id, Bandit.user_id == current_user.id)
        .options(selectinload(Bandit.arms), selectinload(Bandit.state))
    )
    bandit = result.scalar_one_or_none()
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    if not bandit.state:
        raise HTTPException(status_code=400, detail="Bandit has no state initialized")

    if not bandit.arms:
        raise HTTPException(status_code=400, detail="Bandit has no arms configured")

    # Convert to schema objects
    arms = [BanditArmRead.model_validate(arm) for arm in bandit.arms]
    state = BanditStateInternal.from_db(bandit.state)

    # Get analysis
    analysis = get_state_analysis(arms, state)

    if "error" in analysis:
        raise HTTPException(status_code=500, detail=analysis["error"])

    return StateAnalysisResponse(
        bandit_id=bandit_id,
        models=analysis["models"],
        prompts=analysis["prompts"],
        summary=analysis["summary"]
    )


@router.get("/{bandit_id}/budget", response_model=BudgetResponse, tags=["bandit_analysis"])
async def get_budget(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get budget status for a bandit.

    Returns current spend, budget limit, and whether budget is exceeded.
    If no budget is set, is_over_budget will always be False.
    """
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    budget_status = await calculate_budget_status(session, bandit_id, bandit.budget)

    return BudgetResponse(
        bandit_id=bandit_id,
        budget=bandit.budget,
        current_spend=budget_status["current_spend"],
        budget_remaining=budget_status["budget_remaining"],
        budget_used_percent=budget_status["budget_used_percent"],
        is_over_budget=budget_status["is_over_budget"]
    )

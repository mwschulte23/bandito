from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.core.deps import get_current_user
from app.models.user import User
from app.models.bandit import BanditEvent
from app.schemas.bandit_analysis import (
    LeaderboardResponse, StateAnalysisResponse, BudgetResponse,
    ForecastResponse, RegretResponse,
    ArmSummaryResponse, BanditSummaryResponse, BudgetSummary, ConvergenceSummary,
)
from app.services.bandit.bandit_analysis import (
    get_event_leaderboard, get_state_analysis, compute_forecast, compute_regret,
    compute_arm_summary, compute_convergence,
)
from app.services.bandit.helpers import (
    get_bandit_for_user, calculate_budget_status, get_bandit_with_context,
)


router = APIRouter()


@router.get("/{bandit_id}/leaderboard", response_model=LeaderboardResponse, tags=["bandit_analysis"])
async def get_leaderboard(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get enriched leaderboard for a bandit.

    Sorted by adjusted human reward (nulls last, fallback to immediate).
    Includes reward breakdowns (raw, cost_factor, latency_factor, adjusted)
    and Bayesian confidence per arm from the Thompson Sampling posterior.
    """
    ctx = await get_bandit_with_context(bandit_id, session, current_user.id)

    leaderboard = await get_event_leaderboard(
        session, bandit_id, ctx.bandit.arms, ctx.bandit.state, ctx.context
    )

    return LeaderboardResponse(**leaderboard)


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
    ctx = await get_bandit_with_context(bandit_id, session, current_user.id)
    ctx.require_state_and_arms()

    analysis = get_state_analysis(ctx.arms, ctx.state)

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
        remaining=budget_status["remaining"],
        used_percent=budget_status["used_percent"],
        is_over_budget=budget_status["is_over_budget"]
    )


@router.get("/{bandit_id}/forecast", response_model=ForecastResponse, tags=["bandit_analysis"])
async def get_forecast(
    bandit_id: int,
    hour: Optional[int] = Query(None, ge=0, le=23, description="Hour of day (0-23), defaults to current"),
    is_weekend: Optional[int] = Query(None, ge=0, le=1, description="Weekend flag (0 or 1), defaults to current"),
    n_simulations: int = Query(10000, ge=100, le=100000, description="Number of Monte Carlo simulations"),
    beta: float = Query(1.0, ge=0.0, description="Exploration parameter"),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Forecast arm selection probabilities for a given context.

    Uses Monte Carlo simulation over the Thompson Sampling posterior to estimate
    how likely each arm is to be selected under the current model state.

    Returns:
        - selection_probability: P(this arm wins the Thompson sample)
        - expected_score: Mean predicted reward (features @ theta_hat)
        - score_std: Uncertainty in the score prediction
    """
    ctx = await get_bandit_with_context(bandit_id, session, current_user.id)
    ctx.require_state_and_arms()

    # Override context with explicit params if provided
    if hour is not None:
        ctx.context["hour_of_day"] = hour
    if is_weekend is not None:
        ctx.context["is_weekend"] = is_weekend

    forecast = compute_forecast(ctx.arms, ctx.state, ctx.context, n_simulations, beta)

    return ForecastResponse(
        bandit_id=bandit_id,
        context=ctx.context,
        n_simulations=n_simulations,
        avg_uncertainty=forecast["avg_uncertainty"],
        confidence_note=forecast["confidence_note"],
        arms=forecast["arms"]
    )


@router.get("/{bandit_id}/regret", response_model=RegretResponse, tags=["bandit_analysis"])
async def get_regret(
    bandit_id: int,
    window_size: Optional[int] = Query(None, ge=1, description="Events per window. Default: ~20 windows"),
    min_human_events: int = Query(3, ge=1, description="Minimum human-rated events per window for confidence"),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Get regret analysis for a bandit.

    Computes % regret against a global oracle (best active arm) across
    windowed event chunks. Returns both immediate and human reward variants.

    Regret measures how much reward was "left on the table" vs always picking
    the best arm — a decreasing trend proves Thompson Sampling is converging.
    """
    bandit = await get_bandit_for_user(session, bandit_id, current_user.id)
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    try:
        result = await compute_regret(
            session=session,
            bandit_id=bandit_id,
            user_id=current_user.id,
            window_size=window_size,
            min_human_events=min_human_events,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@router.get("/{bandit_id}/arms/summary", response_model=ArmSummaryResponse, tags=["bandit_analysis"])
async def get_arms_summary(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Lightweight arm overview: pull count and preference rank per arm.

    Preference is ranked by expected score (features @ theta_hat), where 1 = best.
    Inactive arms have preference=null.
    """
    ctx = await get_bandit_with_context(bandit_id, session, current_user.id)
    ctx.require_state_and_arms()

    # Pull counts per arm
    counts_result = await session.execute(
        select(BanditEvent.arm_id, func.count())
        .where(BanditEvent.bandit_id == bandit_id)
        .group_by(BanditEvent.arm_id)
    )
    pull_counts = dict(counts_result.all())

    entries = compute_arm_summary(ctx.arms, ctx.state, ctx.context, pull_counts)

    return ArmSummaryResponse(bandit_id=bandit_id, arms=entries)


@router.get("/{bandit_id}/summary", response_model=BanditSummaryResponse, tags=["bandit_analysis"])
async def get_bandit_summary(
    bandit_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    High-level bandit overview: total pulls, budget usage, and convergence.

    Convergence is the average posterior uncertainty across active arms,
    mapped to high/medium/low confidence labels.
    """
    ctx = await get_bandit_with_context(bandit_id, session, current_user.id)

    # Total pulls
    count_result = await session.execute(
        select(func.count())
        .where(BanditEvent.bandit_id == bandit_id)
    )
    total_pulls = count_result.scalar()

    # Budget
    budget_status = await calculate_budget_status(session, bandit_id, ctx.bandit.budget)
    budget = BudgetSummary(
        budget=ctx.bandit.budget,
        current_spend=budget_status["current_spend"],
        used_percent=budget_status["used_percent"],
        is_over_budget=budget_status["is_over_budget"],
    )

    # Convergence
    confidence_val, label = (1.0, "low")
    if ctx.state and ctx.arms:
        confidence_val, label = compute_convergence(ctx.arms, ctx.state, ctx.context)

    convergence = ConvergenceSummary(confidence=confidence_val, label=label)

    return BanditSummaryResponse(
        bandit_id=bandit_id,
        total_pulls=total_pulls,
        budget=budget,
        convergence=convergence,
    )

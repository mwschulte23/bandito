from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.bandit import Bandit, BanditArm, BanditEvent
from app.schemas.bandit import BanditArmRead, BanditStateInternal
from app.services.bandit.utils.feature_prep import compute_feature_dimensions


@dataclass
class BanditContext:
    bandit: Bandit
    arms: list[BanditArmRead]
    state: Optional[BanditStateInternal]
    context: dict  # {"hour_of_day": int, "is_weekend": int}

    def require_state_and_arms(self):
        if not self.state:
            raise HTTPException(400, "Bandit has no state initialized")
        if not self.arms:
            raise HTTPException(400, "Bandit has no arms configured")


async def get_bandit_with_context(
    bandit_id: int, session: AsyncSession, user_id: int
) -> BanditContext:
    """Load bandit with arms + state, convert to schemas, build time context."""
    result = await session.execute(
        select(Bandit)
        .where(Bandit.id == bandit_id, Bandit.user_id == user_id)
        .options(selectinload(Bandit.arms), selectinload(Bandit.state))
    )
    bandit = result.scalar_one_or_none()
    if not bandit:
        raise HTTPException(status_code=404, detail="Bandit not found")

    arms = [BanditArmRead.model_validate(arm) for arm in bandit.arms]
    state = BanditStateInternal.from_db(bandit.state) if bandit.state else None

    now = datetime.now(timezone.utc)
    context = {"hour_of_day": now.hour, "is_weekend": 1 if now.weekday() >= 5 else 0}

    return BanditContext(bandit=bandit, arms=arms, state=state, context=context)


async def get_bandit_for_user(
    session: AsyncSession, bandit_id: int, user_id: int
) -> Bandit | None:
    """Fetch a bandit with ownership verification."""
    result = await session.execute(
        select(Bandit).where(Bandit.id == bandit_id, Bandit.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def calculate_bandit_dimensions(session: AsyncSession, bandit_id: int) -> int:
    """
    Calculate feature dimensions for a bandit based on its arms.
    """
    result = await session.execute(
        select(BanditArm).where(BanditArm.bandit_id == bandit_id)
    )
    arms = result.scalars().all()

    models = set(arm.model_name for arm in arms)
    prompts = set(arm.system_prompt for arm in arms)

    return compute_feature_dimensions(len(models), len(prompts))


async def calculate_budget_status(
    session: AsyncSession, bandit_id: int, budget: float | None
) -> dict:
    """
    Calculate budget status for a bandit.

    Returns:
        dict with keys: current_spend, remaining, used_percent,
                       is_over_budget, budget_warning
    """
    result = await session.execute(
        select(func.coalesce(func.sum(BanditEvent.cost), 0))
        .where(BanditEvent.bandit_id == bandit_id)
    )
    current_spend = float(result.scalar())

    if budget is None or budget <= 0:
        return {
            "current_spend": round(current_spend, 4),
            "remaining": None,
            "used_percent": None,
            "is_over_budget": False,
            "budget_warning": None,
        }

    remaining = budget - current_spend
    used_percent = current_spend / budget

    budget_warning = None
    if used_percent >= 0.90:
        budget_warning = f"{used_percent * 100:.1f}% of budget consumed (${current_spend:.2f} / ${budget:.2f})"

    return {
        "current_spend": round(current_spend, 4),
        "remaining": round(remaining, 4),
        "used_percent": round(used_percent, 2),
        "is_over_budget": current_spend >= budget,
        "budget_warning": budget_warning,
    }

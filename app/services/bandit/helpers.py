from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.bandit import Bandit, BanditArm, BanditEvent
from app.services.bandit.utils.feature_prep import compute_feature_dimensions


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
        dict with keys: current_spend, budget_remaining, budget_used_percent,
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
            "budget_remaining": None,
            "budget_used_percent": None,
            "is_over_budget": False,
            "budget_warning": None,
        }

    budget_remaining = budget - current_spend
    budget_used_percent = current_spend / budget

    budget_warning = None
    if budget_used_percent >= 0.90:
        budget_warning = f"{budget_used_percent * 100:.1f}% of budget consumed (${current_spend:.2f} / ${budget:.2f})"

    return {
        "current_spend": round(current_spend, 4),
        "budget_remaining": round(budget_remaining, 4),
        "budget_used_percent": round(budget_used_percent, 2),
        "is_over_budget": current_spend >= budget,
        "budget_warning": budget_warning,
    }

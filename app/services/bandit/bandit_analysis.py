import numpy as np
from typing import Dict, List, Any
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.bandit import BanditArm, BanditEvent
from app.schemas.bandit import BanditArmRead, BanditStateInternal
from app.services.bandit.utils.feature_prep import FeatureTransformer


async def get_event_leaderboard(
    session: AsyncSession,
    bandit_id: int
) -> List[Dict[str, Any]]:
    """
    Aggregate event-level stats per arm.
    Returns pull count, average rewards, cost, latency.
    """
    # Get all arms for this bandit
    arms_result = await session.execute(
        select(BanditArm).where(BanditArm.bandit_id == bandit_id)
    )
    arms = {arm.id: arm for arm in arms_result.scalars().all()}

    # Aggregate events per arm
    stats_query = (
        select(
            BanditEvent.arm_id,
            func.count(BanditEvent.id).label('pull_count'),
            func.avg(BanditEvent.immediate_reward).label('avg_immediate_reward'),
            func.avg(BanditEvent.human_reward).label('avg_human_reward'),
            func.avg(BanditEvent.cost).label('avg_cost'),
            func.sum(BanditEvent.cost).label('total_cost'),
            func.avg(BanditEvent.latency).label('avg_latency'),
            func.sum(BanditEvent.latency).label('total_latency'),
            func.min(BanditEvent.created_at).label('first_pull'),
            func.max(BanditEvent.created_at).label('last_pull'),
        )
        .where(BanditEvent.bandit_id == bandit_id)
        .group_by(BanditEvent.arm_id)
    )

    stats_result = await session.execute(stats_query)
    rows = stats_result.all()

    leaderboard = []
    for row in rows:
        arm = arms.get(row.arm_id)
        if not arm:
            continue

        avg_cost = float(row.avg_cost or 0) if row.avg_cost else None
        total_cost = float(row.total_cost or 0) if row.total_cost else None
        avg_latency = float(row.avg_latency or 0) if row.avg_latency else None
        total_latency = float(row.total_latency or 0) if row.total_latency else None

        leaderboard.append({
            "arm_id": row.arm_id,
            "model_name": arm.model_name,
            "system_prompt": arm.system_prompt[:50] + "..." if len(arm.system_prompt) > 50 else arm.system_prompt,
            "is_active": arm.is_active,
            "pull_count": row.pull_count,
            "avg_immediate_reward": round(float(row.avg_immediate_reward or 0), 4),
            "avg_human_reward": round(float(row.avg_human_reward or 0), 4) if row.avg_human_reward else None,
            # Raw values
            "avg_cost": round(avg_cost, 4) if avg_cost else None,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
            "total_cost": round(total_cost, 4) if total_cost else None,
            "total_latency_ms": round(total_latency, 1) if total_latency else None,
            # Human-friendly display
            "avg_cost_display": f"${avg_cost:.4f}" if avg_cost else None,
            "total_cost_display": f"${total_cost:.2f}" if total_cost else None,
            "avg_latency_display": _format_latency(avg_latency) if avg_latency else None,
            "total_latency_display": _format_latency(total_latency) if total_latency else None,
            "first_pull": row.first_pull.isoformat() if row.first_pull else None,
            "last_pull": row.last_pull.isoformat() if row.last_pull else None,
        })

    # Sort by average immediate reward descending
    leaderboard.sort(key=lambda x: x["avg_immediate_reward"], reverse=True)

    return leaderboard


def _format_latency(ms: float) -> str:
    """Format latency in human-readable form."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def get_state_analysis(
    arms: List[BanditArmRead],
    state: BanditStateInternal
) -> Dict[str, Any]:
    """
    Decompose theta_hat into interpretable model/prompt/time effects.
    Shows weights and uncertainty per component.
    """
    mapper = FeatureTransformer(arms)
    names = mapper.get_feature_names()

    theta = state.theta_hat
    A = state.a

    # Validate dimensions match
    if len(theta) != len(names):
        return {"error": f"Dimension mismatch: theta={len(theta)}, features={len(names)}"}

    # Compute uncertainty from A_inv diagonal
    try:
        A_inv_diag = np.diag(np.linalg.inv(A))
    except np.linalg.LinAlgError:
        A_inv_diag = np.ones(len(names)) * float('inf')

    A_diag = np.diag(A)  # Data volume indicator

    analysis = {
        "models": {},
        "prompts": {},
        "summary": {}
    }

    # Group features by model
    for model in mapper.models:
        model_features = {}
        total_effect = 0.0

        for i, name in enumerate(names):
            if f"model_{model}" in name:
                # Clean up feature name
                short_name = name.replace(f"_x_model_{model}", "").replace(f"model_{model}", "baseline")

                weight = float(theta[i])
                model_features[short_name] = {
                    "weight": round(weight, 4),
                    "uncertainty": round(float(A_inv_diag[i]), 6),
                    "observations": round(float(A_diag[i]), 1),
                }

                # Baseline contributes directly to total effect
                if short_name == "baseline":
                    total_effect += weight

        analysis["models"][model] = {
            "features": model_features,
            "baseline_effect": round(total_effect, 4),
        }

    # Extract prompt effects (shared across models)
    for prompt in mapper.prompts:
        prompt_key = f"prompt_{prompt}"
        for i, name in enumerate(names):
            if name == prompt_key:
                analysis["prompts"][prompt[:30] + "..." if len(prompt) > 30 else prompt] = {
                    "weight": round(float(theta[i]), 4),
                    "uncertainty": round(float(A_inv_diag[i]), 6),
                    "observations": round(float(A_diag[i]), 1),
                }
                break

    # Summary stats
    total_observations = float(np.sum(A_diag)) / len(names)  # Average across features
    avg_uncertainty = float(np.mean(A_inv_diag[np.isfinite(A_inv_diag)]))

    # Rank models by baseline effect
    model_ranking = sorted(
        [(m, data["baseline_effect"]) for m, data in analysis["models"].items()],
        key=lambda x: x[1],
        reverse=True
    )

    analysis["summary"] = {
        "total_features": len(names),
        "avg_observations_per_feature": round(total_observations, 1),
        "avg_uncertainty": round(avg_uncertainty, 6),
        "model_ranking": [{"model": m, "baseline_effect": e} for m, e in model_ranking],
    }
    
    return analysis

import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.bandit import Bandit, BanditArm, BanditEvent
from app.schemas.bandit import BanditArmRead, BanditStateInternal
from app.schemas.bandit_analysis import (
    RegretMetric, RegretWindow, RegretResponse,
    RewardBreakdown, ArmLeaderboardEntry, LeaderboardResponse,
    ArmSummaryEntry,
)
from app.services.bandit.utils.feature_prep import FeatureTransformer
from app.services.bandit.utils.rewards import calculate_reward, importance_to_sensitivity


def compute_penalty_factors(event: BanditEvent) -> tuple:
    """Returns (cost_factor, latency_factor) for an event."""
    cost_imp = event.cost_importance if event.cost_importance is not None else 2
    lat_imp = event.latency_importance if event.latency_importance is not None else 2
    max_cost, max_latency = 5.0, 60000.0

    if event.cost is not None and event.cost > max_cost:
        cost_f = 0.0
    elif event.cost is not None:
        cost_f = float(np.exp(-importance_to_sensitivity(cost_imp) * event.cost / max_cost))
    else:
        cost_f = 1.0

    if event.latency is not None and event.latency > max_latency:
        lat_f = 0.0
    elif event.latency is not None:
        lat_f = float(np.exp(-importance_to_sensitivity(lat_imp) * event.latency / max_latency))
    else:
        lat_f = 1.0

    return cost_f, lat_f


def _confidence_label(score_std: float) -> str:
    if score_std < 0.1:
        return "high"
    elif score_std < 0.3:
        return "medium"
    else:
        return "low"


def compute_arm_means(
    events: List[BanditEvent],
    arm_ids: set,
    reward_fn: Callable[[BanditEvent], Optional[float]],
) -> Dict[int, tuple]:
    """
    Group events by arm, apply reward_fn, return {arm_id: (mean, count)}.
    Skips events where reward_fn returns None.
    Only includes arms in arm_ids.
    """
    arm_rewards: Dict[int, List[float]] = defaultdict(list)
    for event in events:
        if event.arm_id not in arm_ids:
            continue
        val = reward_fn(event)
        if val is not None:
            arm_rewards[event.arm_id].append(val)

    return {
        aid: (float(np.mean(vals)), len(vals))
        for aid, vals in arm_rewards.items()
        if vals
    }


def _build_reward_breakdown(
    events: List[BanditEvent],
    reward_attr: str,
) -> Optional[RewardBreakdown]:
    """Build a RewardBreakdown from events using the given reward attribute."""
    valid = []
    for e in events:
        reward_val = getattr(e, reward_attr, None)
        if reward_val is None:
            continue
        cost_f, lat_f = compute_penalty_factors(e)
        valid.append((reward_val, cost_f, lat_f, reward_val * cost_f * lat_f))

    if not valid:
        return None

    raws, cost_fs, lat_fs, adjusteds = zip(*valid)
    return RewardBreakdown(
        raw=round(float(np.mean(raws)), 4),
        cost_factor=round(float(np.mean(cost_fs)), 4),
        latency_factor=round(float(np.mean(lat_fs)), 4),
        adjusted=round(float(np.mean(adjusteds)), 4),
    )


async def get_event_leaderboard(
    session: AsyncSession,
    bandit_id: int,
    arms: List[BanditArm],
    state: Optional[Any],
    context: dict,
) -> dict:
    """
    Compute enriched leaderboard with reward breakdowns and confidence.
    Returns dict ready to unpack into LeaderboardResponse.
    """
    # Load all events for this bandit
    events_result = await session.execute(
        select(BanditEvent)
        .where(BanditEvent.bandit_id == bandit_id)
        .order_by(BanditEvent.created_at)
    )
    all_events = list(events_result.scalars().all())

    # Group events by arm
    arm_events: Dict[int, List[BanditEvent]] = defaultdict(list)
    for event in all_events:
        arm_events[event.arm_id].append(event)

    # Precompute confidence: A_inv and feature vectors
    arm_confidence: Dict[int, float] = {}
    if state is not None:
        arms_schema = [BanditArmRead.model_validate(arm) for arm in arms]
        if not isinstance(state, BanditStateInternal):
            state_internal = BanditStateInternal.from_db(state)
        else:
            state_internal = state

        mapper = FeatureTransformer(arms_schema)
        A_inv = np.linalg.inv(state_internal.a)

        for arm_schema in arms_schema:
            features = mapper.transform_to_vector(arm_schema, context)
            score_std = float(np.sqrt(features @ A_inv @ features))
            arm_confidence[arm_schema.id] = score_std

    # Build entries
    entries = []
    total_human = 0

    for arm in arms:
        arm_evts = arm_events.get(arm.id, [])
        pull_count = len(arm_evts)

        # Human and immediate breakdowns
        human_breakdown = _build_reward_breakdown(arm_evts, "human_reward")
        immediate_breakdown = _build_reward_breakdown(arm_evts, "immediate_reward")

        # If no immediate events, provide a zero breakdown
        if immediate_breakdown is None:
            immediate_breakdown = RewardBreakdown(raw=0.0, cost_factor=None, latency_factor=None, adjusted=0.0)

        # Human count
        h_count = sum(1 for e in arm_evts if e.human_reward is not None)
        total_human += h_count

        # Confidence
        score_std = arm_confidence.get(arm.id, 1.0)

        # Avg cost / latency
        costs = [e.cost for e in arm_evts if e.cost is not None]
        latencies = [e.latency for e in arm_evts if e.latency is not None]

        entries.append(ArmLeaderboardEntry(
            arm_id=arm.id,
            model_name=arm.model_name,
            system_prompt=arm.system_prompt,
            is_active=arm.is_active,
            human=human_breakdown,
            immediate=immediate_breakdown,
            pull_count=pull_count,
            human_count=h_count,
            confidence=round(score_std, 4),
            confidence_label=_confidence_label(score_std),
            avg_cost=round(float(np.mean(costs)), 4) if costs else None,
            avg_latency_ms=round(float(np.mean(latencies)), 1) if latencies else None,
        ))

    # Sort: human.adjusted desc (nulls last), fallback to immediate.adjusted
    def sort_key(entry: ArmLeaderboardEntry):
        if entry.human is not None and entry.human.adjusted is not None:
            return (1, entry.human.adjusted)
        if entry.immediate.adjusted is not None:
            return (0, entry.immediate.adjusted)
        return (-1, 0)

    entries.sort(key=sort_key, reverse=True)

    # Overall confidence: average score_std across active arms
    active_stds = [e.confidence for e in entries if e.is_active]
    overall_conf = float(np.mean(active_stds)) if active_stds else 1.0

    return {
        "bandit_id": bandit_id,
        "total_pulls": len(all_events),
        "total_human": total_human,
        "overall_confidence": round(overall_conf, 4),
        "overall_confidence_label": _confidence_label(overall_conf),
        "arms": entries,
    }


def compute_forecast(
    arms: List[BanditArmRead],
    state: BanditStateInternal,
    context: dict,
    n_simulations: int = 10000,
    beta: float = 1.0
) -> Dict[str, Any]:
    """Monte Carlo estimation of arm selection probabilities."""
    mapper = FeatureTransformer(arms)
    active_arms = [a for a in arms if a.is_active]

    if not active_arms:
        return {"arms": [], "avg_uncertainty": 0.0, "confidence_note": "No active arms"}

    # Get features for each arm
    arm_features = {a.id: mapper.transform_to_vector(a, context) for a in active_arms}

    # Precompute A_inv for uncertainty calculation
    A_inv = np.linalg.inv(state.a)

    # Monte Carlo: sample theta, compute scores, count wins
    wins = {a.id: 0 for a in active_arms}
    for _ in range(n_simulations):
        theta_sample = state.theta_hat + beta * (state.cholesky_l_inv @ np.random.standard_normal(state.dimensions))
        scores = {aid: features @ theta_sample for aid, features in arm_features.items()}
        winner = max(scores, key=scores.get)
        wins[winner] += 1

    # Build per-arm results
    results = []
    for arm in active_arms:
        features = arm_features[arm.id]
        results.append({
            "arm_id": arm.id,
            "model_name": arm.model_name,
            "system_prompt": arm.system_prompt,
            "selection_probability": wins[arm.id] / n_simulations,
            "expected_score": float(features @ state.theta_hat),
            "score_std": float(np.sqrt(features @ A_inv @ features))
        })

    results = sorted(results, key=lambda x: -x["selection_probability"])

    # Compute average uncertainty
    avg_uncertainty = np.mean([r["score_std"] for r in results])

    # Generate confidence note based on uncertainty level and selection concentration
    top_prob = results[0]["selection_probability"] if results else 0
    confidence_note = _generate_confidence_note(avg_uncertainty, top_prob, len(results))

    return {
        "arms": results,
        "avg_uncertainty": round(float(avg_uncertainty), 4),
        "confidence_note": confidence_note
    }


def _generate_confidence_note(avg_uncertainty: float, top_prob: float, n_arms: int) -> str:
    """Generate human-readable interpretation of forecast confidence."""
    # Uniform distribution baseline
    uniform_prob = 1.0 / n_arms if n_arms > 0 else 0

    # How concentrated are selections vs uniform?
    concentration = top_prob / uniform_prob if uniform_prob > 0 else 1.0

    if avg_uncertainty > 0.5:
        if concentration < 1.5:
            return "High uncertainty: posteriors overlap significantly. Selection probabilities are exploratory, not indicative of true arm quality."
        else:
            return "High uncertainty but emerging preference. Early signal favors top arm, but confidence intervals still overlap."
    elif avg_uncertainty > 0.2:
        if concentration < 2.0:
            return "Moderate uncertainty: arms are competitive. More data needed to distinguish performance."
        else:
            return "Moderate uncertainty with clear leader. Top arm shows consistent advantage, though some exploration continues."
    else:
        if concentration < 2.0:
            return "Low uncertainty: arms have similar true performance. Selection reflects genuine competitive parity."
        else:
            return "Low uncertainty: selection probabilities reflect learned preferences with high confidence."


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
                analysis["prompts"][prompt[:50] + "..." if len(prompt) > 50 else prompt] = {
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


# ============ Regret Analysis ============

def compute_oracle(
    events: List[BanditEvent],
    active_arm_ids: set[int],
    reward_fn: Callable[[BanditEvent], Optional[float]],
) -> Optional[tuple[int, float]]:
    """
    Find the best arm (oracle) by mean reward across all events.

    Groups events by arm_id (active arms only), applies reward_fn, skips Nones.
    Returns (best_arm_id, best_arm_mean) or None if no valid events.
    """
    arm_means = compute_arm_means(events, active_arm_ids, reward_fn)
    if not arm_means:
        return None
    best = max(arm_means, key=lambda aid: arm_means[aid][0])
    return best, arm_means[best][0]


def compute_window_regret(
    window_events: List[BanditEvent],
    oracle_mean: float,
    oracle_arm_id: int,
    reward_fn: Callable[[BanditEvent], Optional[float]],
) -> Optional[RegretMetric]:
    """
    Compute regret for a single window of events against a global oracle.

    Returns None if no valid events or oracle_mean is 0.
    """
    chosen_rewards = []
    for event in window_events:
        val = reward_fn(event)
        if val is not None:
            chosen_rewards.append(val)

    if not chosen_rewards or oracle_mean == 0:
        return None

    n = len(chosen_rewards)
    sum_optimal = oracle_mean * n
    sum_chosen = sum(chosen_rewards)
    mean_regret = oracle_mean - (sum_chosen / n)
    pct = (sum_optimal - sum_chosen) / sum_optimal * 100

    return RegretMetric(
        pct=round(pct, 4),
        mean_regret=round(mean_regret, 6),
        mean_optimal=round(oracle_mean, 6),
        best_arm_id=oracle_arm_id,
    )


async def compute_regret(
    session: AsyncSession,
    bandit_id: int,
    user_id: int,
    window_size: Optional[int] = None,
    min_human_events: int = 3,
) -> dict:
    """
    Orchestrate full regret analysis for a bandit.

    Loads all events, computes oracle per metric, windows events,
    and returns structured regret data.
    """
    # Load bandit with arms
    result = await session.execute(
        select(Bandit)
        .where(Bandit.id == bandit_id, Bandit.user_id == user_id)
        .options(selectinload(Bandit.arms))
    )
    bandit = result.scalar_one_or_none()
    if not bandit:
        raise ValueError("Bandit not found")

    # Load ALL events ordered by created_at
    events_result = await session.execute(
        select(BanditEvent)
        .where(BanditEvent.bandit_id == bandit_id)
        .order_by(BanditEvent.created_at)
    )
    events = list(events_result.scalars().all())

    if not events:
        raise ValueError("Bandit has no events")

    # Active arm IDs
    active_arm_ids = {arm.id for arm in bandit.arms if arm.is_active}
    if not active_arm_ids:
        raise ValueError("Bandit has no active arms")

    # Define reward functions
    def accuracy_fn(e: BanditEvent) -> Optional[float]:
        return e.immediate_reward

    def adjusted_fn(e: BanditEvent) -> Optional[float]:
        if e.immediate_reward is None:
            return None
        return calculate_reward(
            reward=e.immediate_reward,
            cost=e.cost,
            latency=e.latency,
            cost_importance=e.cost_importance if e.cost_importance is not None else 2,
            latency_importance=e.latency_importance if e.latency_importance is not None else 2,
        )

    def cost_fn(e: BanditEvent) -> Optional[float]:
        if e.cost is None:
            return None
        return calculate_reward(
            reward=1.0,
            cost=e.cost,
            latency=None,
            cost_importance=e.cost_importance if e.cost_importance is not None else 2,
            latency_importance=0,
        )

    def latency_fn(e: BanditEvent) -> Optional[float]:
        if e.latency is None:
            return None
        return calculate_reward(
            reward=1.0,
            cost=None,
            latency=e.latency,
            cost_importance=0,
            latency_importance=e.latency_importance if e.latency_importance is not None else 2,
        )

    def human_accuracy_fn(e: BanditEvent) -> Optional[float]:
        return e.human_reward

    def human_adjusted_fn(e: BanditEvent) -> Optional[float]:
        if e.human_reward is None:
            return None
        return calculate_reward(
            reward=e.human_reward,
            cost=e.cost,
            latency=e.latency,
            cost_importance=e.cost_importance if e.cost_importance is not None else 2,
            latency_importance=e.latency_importance if e.latency_importance is not None else 2,
        )

    metric_fns = {"accuracy": accuracy_fn, "adjusted": adjusted_fn, "cost": cost_fn, "latency": latency_fn}
    human_metric_fns = {"accuracy": human_accuracy_fn, "adjusted": human_adjusted_fn}

    # Compute global oracles
    oracles: Dict[str, Optional[tuple[int, float]]] = {}
    for name, fn in metric_fns.items():
        oracles[name] = compute_oracle(events, active_arm_ids, fn)

    human_oracles: Dict[str, Optional[tuple[int, float]]] = {}
    for name, fn in human_metric_fns.items():
        human_oracles[name] = compute_oracle(events, active_arm_ids, fn)

    # Default window_size: ~20 windows
    if window_size is None:
        window_size = max(10, len(events) // 20)

    # Chunk events into windows
    windows_data: list[RegretWindow] = []
    for i in range(0, len(events), window_size):
        chunk = events[i:i + window_size]
        window_idx = i // window_size

        # Compute metrics for this window
        metrics: Dict[str, RegretMetric] = {}
        for name, fn in metric_fns.items():
            oracle = oracles[name]
            if oracle is not None:
                result_metric = compute_window_regret(chunk, oracle[1], oracle[0], fn)
                if result_metric is not None:
                    metrics[name] = result_metric

        # Human metrics
        human_metrics: Dict[str, Optional[RegretMetric]] = {}
        human_events_in_window = [e for e in chunk if e.human_reward is not None]
        human_n = len(human_events_in_window)

        for name, fn in human_metric_fns.items():
            oracle = human_oracles[name]
            if oracle is not None:
                result_metric = compute_window_regret(chunk, oracle[1], oracle[0], fn)
                human_metrics[name] = result_metric
            else:
                human_metrics[name] = None

        windows_data.append(RegretWindow(
            window=window_idx,
            event_range=(chunk[0].id, chunk[-1].id),
            n_events=len(chunk),
            metrics=metrics,
            human_metrics=human_metrics,
            human_n_events=human_n,
            human_low_confidence=human_n < min_human_events,
        ))

    # Compute overall (all events as one big window)
    overall: Dict[str, RegretMetric] = {}
    for name, fn in metric_fns.items():
        oracle = oracles[name]
        if oracle is not None:
            result_metric = compute_window_regret(events, oracle[1], oracle[0], fn)
            if result_metric is not None:
                overall[name] = result_metric

    human_overall: Dict[str, Optional[RegretMetric]] = {}
    for name, fn in human_metric_fns.items():
        oracle = human_oracles[name]
        if oracle is not None:
            result_metric = compute_window_regret(events, oracle[1], oracle[0], fn)
            human_overall[name] = result_metric
        else:
            human_overall[name] = None

    return RegretResponse(
        bandit_id=bandit_id,
        total_events=len(events),
        window_size=window_size,
        windows=windows_data,
        overall=overall,
        human_overall=human_overall,
    )


# ============ Arm Summary ============

def compute_arm_summary(
    arms: List[BanditArmRead],
    state: BanditStateInternal,
    context: dict,
    pull_counts: Dict[int, int],
) -> list[ArmSummaryEntry]:
    """
    Build arm summary with pull counts and preference rank from theta_hat.
    Rank is based on expected score (features @ theta_hat), 1 = best.
    Inactive arms get preference=None.
    """
    mapper = FeatureTransformer(arms)

    # Score active arms for ranking
    active_scores = []
    for arm in arms:
        if arm.is_active:
            features = mapper.transform_to_vector(arm, context)
            score = float(features @ state.theta_hat)
            active_scores.append((arm.id, score))

    # Rank: highest score = rank 1
    active_scores.sort(key=lambda x: x[1], reverse=True)
    rank_map = {arm_id: rank + 1 for rank, (arm_id, _) in enumerate(active_scores)}

    entries = []
    for arm in arms:
        entries.append(ArmSummaryEntry(
            arm_id=arm.id,
            model_name=arm.model_name,
            system_prompt=arm.system_prompt,
            is_active=arm.is_active,
            pull_count=pull_counts.get(arm.id, 0),
            preference=rank_map.get(arm.id),
        ))

    return entries


def compute_convergence(
    arms: List[BanditArmRead],
    state: BanditStateInternal,
    context: dict,
) -> tuple[float, str]:
    """
    Average posterior uncertainty across active arms → (confidence, label).
    """
    active_arms = [a for a in arms if a.is_active]
    if not active_arms:
        return 1.0, _confidence_label(1.0)

    mapper = FeatureTransformer(arms)
    A_inv = np.linalg.inv(state.a)

    stds = []
    for arm in active_arms:
        features = mapper.transform_to_vector(arm, context)
        stds.append(float(np.sqrt(features @ A_inv @ features)))

    confidence_val = round(float(np.mean(stds)), 4)
    return confidence_val, _confidence_label(confidence_val)

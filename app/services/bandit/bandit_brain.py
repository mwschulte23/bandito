"""
Core bandit algorithm implementation.

Thompson Sampling for contextual bandits with linear payoffs.
"""

import numpy as np
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.bandit import BanditStateInternal
from app.services.bandit.utils.feature_prep import FeatureTransformer
from app.services.bandit.utils.data_helper import get_full_bandit, add_event, get_event, update_event, update_state
from app.services.bandit.utils.rewards import calculate_reward


async def pull_arm(
    session: AsyncSession,
    bandit_id: int,
    user_id: int,
    user_query: str,
    gamma: float = 1.0,
    beta: float = 1.0
) -> dict:
    """
    Select an arm using Thompson Sampling.

    Args:
        session: Database session
        bandit_id: ID of the bandit
        user_id: ID of the user (for ownership verification)
        user_query: The user's query being processed
        gamma: Regularization parameter (unused currently)
        beta: Exploration parameter for Thompson Sampling

    Returns:
        dict with event_id, arm_id, and context
    """
    bandit = await get_full_bandit(session, bandit_id, user_id)

    # Server-computed context from current time (UTC)
    now = datetime.now(timezone.utc)
    context = {
        "hour_of_day": now.hour,
        "is_weekend": 1 if now.weekday() >= 5 else 0,
    }

    mapper = FeatureTransformer(bandit.arms)
    ts_theta = bandit.state.theta_hat + beta * (bandit.state.cholesky_l_inv @ np.random.standard_normal(bandit.state.dimensions))

    best_arm_id, max_score = None, -float('inf')
    for arm in bandit.arms:
        if not arm.is_active:
            continue
        features = mapper.transform_to_vector(arm, context)
        score = features @ ts_theta
        if score > max_score:
            max_score = score
            best_arm_id = arm.id

    event_id = await add_event(session, bandit_id, best_arm_id, max_score, context, user_query)
    return {
        'event_id': event_id,
        'arm_id': best_arm_id,
        'context': context
    }


async def update_on_reward(
    session: AsyncSession,
    bandit_id: int,
    user_id: int,
    event_id: int,
    reward: float,
    cost: float,
    latency: float,
    llm_output: str | dict = None,
    is_human_reward: bool = False
) -> None:
    """
    Update bandit state with reward signal.

    Args:
        session: Database session
        bandit_id: ID of the bandit
        user_id: ID of the user
        event_id: ID of the event to update
        llm_output: LLM response (stored for human review)
        reward: Reward value
        cost: API cost in dollars
        latency: Response time in ms
        is_human_reward: Whether this is human feedback (True) or immediate reward (False)
    """
    bandit = await get_full_bandit(session, bandit_id, user_id)
    target_event = await get_event(session, event_id)
    chosen_arm = [arm for arm in bandit.arms if arm.id == target_event.arm_id][0]
    state = bandit.state

    mapper = FeatureTransformer(bandit.arms)
    features = mapper.transform_to_vector(chosen_arm, target_event.context)
    
    if is_human_reward:
        if target_event.human_reward is not None:
            # PATH 3: Changing existing human reward
            # Delta between new and old adjusted human rewards
            # Only adjusts b (observation count unchanged)
            old_adj = calculate_reward(target_event.human_reward, cost, latency)
            new_adj = calculate_reward(reward, cost, latency)
            state.b += features * (new_adj - old_adj)
        elif target_event.immediate_reward is not None:
            # PATH 2: Normal residual (immediate exists, first human)
            adj_human = calculate_reward(reward, cost, latency)
            adj_immediate = calculate_reward(target_event.immediate_reward, cost, latency)
            state.b += features * (adj_human - adj_immediate)
        else:
            # PATH 1: First reward ever (human before immediate)
            adjusted_reward = calculate_reward(reward, cost, latency)
            state.a += np.outer(features, features)
            state.b += features * adjusted_reward
            target_event.llm_output = llm_output
            target_event.cost = cost
            target_event.latency = latency

        target_event.human_reward = reward  # Always set AFTER reading old value
    else: # treat as first reward applied
        adjusted_reward = calculate_reward(reward, cost, latency)
        state.a += np.outer(features, features)
        state.b += features * adjusted_reward

        target_event.immediate_reward = reward
        target_event.llm_output = llm_output
        target_event.cost = cost
        target_event.latency = latency

    A_inv = np.linalg.inv(state.a)
    state.theta_hat = A_inv @ state.b
    state.cholesky_l_inv = np.linalg.cholesky(A_inv)

    await update_event(session, target_event)
    await update_state(session, state.id, state)

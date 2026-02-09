import numpy as np
from typing import List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.bandit import Bandit, BanditArm, BanditState, BanditEvent
from app.services.bandit.utils.feature_prep import FeatureTransformer

    

def importance_to_sensitivity(importance: int) -> float:
    """Convert user-facing importance (0-5) to internal sensitivity parameter.

    Importance is an intuitive 0-5 scale:
      0 = ignore this factor entirely
      5 = this factor is critical

    Currently a 1:1 mapping. Isolated here so the mapping can evolve
    without touching callers.
    """
    return float(importance)


def calculate_reward(
    reward: float,
    cost: float = None,
    latency: float = None,
    cost_importance: int = 2,
    latency_importance: int = 2,
    max_cost: float = 5.0, # dollars
    max_latency: float = 1000 * 60 # milliseconds
):
    cost_sensitivity = importance_to_sensitivity(cost_importance)
    lat_sensitivity = importance_to_sensitivity(latency_importance)

    if cost is not None:
        if cost > max_cost:
            reward = 0
        else:
            scaled_cost = cost / max_cost
            reward *= np.exp(-cost_sensitivity * scaled_cost)

    if latency is not None:
        if latency > max_latency:
            reward = 0
        else:
            scaled_latency = latency / max_latency
            reward *= np.exp(-lat_sensitivity * scaled_latency)

    return np.clip(reward, 0, 1)



    # immediate_reward: Optional[float] = Field(default=None)
    # human_reward: Optional[float] = Field(default=None)
    # outcome_reward: Optional[float] = Field(default=None)
    # cost: Optional[float] = Field(default=None)
    # latency: Optional[float] = Field(default=None)


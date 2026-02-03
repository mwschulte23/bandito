import numpy as np
from typing import List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.bandit import Bandit, BanditArm, BanditState, BanditEvent
from app.services.bandit.utils.feature_prep import FeatureTransformer

    

def calculate_reward(
    reward: float,
    cost: float = None,
    latency: float = None,
    cost_sensitivity: float = 2.0,
    lat_sensitivity: float = 2.0,
    max_cost: float = 5.0, # dollars
    max_latency: float = 1000 * 60 # milliseconds
):
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


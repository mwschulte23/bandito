import io
import numpy as np
from typing import Optional, List, Dict, Literal
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class RewardInput(BaseModel):
    score: float
    type: Literal['immediate', 'human', 'outcome']
    apply_overhead: bool = Field(default=True)
    cost: Optional[float] = Field(default=None)
    latency: Optional[float] = Field(default=None)
from typing import Optional, Dict, Literal
from pydantic import BaseModel, Field

from app.schemas.bandit import BanditArmRead


class PullArmRequest(BaseModel):
    user_query: str
    gamma: float = Field(default=1.0, description="Regularization parameter")
    beta: float = Field(default=1.0, description="Exploration parameter")


class PullArmResponse(BaseModel):
    event_id: int
    arm: BanditArmRead
    context: Dict
    budget_warning: Optional[str] = None


class ImmediateRewardRequest(BaseModel):
    """Immediate reward submitted right after LLM response."""
    score: float = Field(ge=0, le=1, description="Reward value [0, 1]")
    llm_output: str | Dict = Field(description="LLM response for human review")
    cost: float = Field(ge=0, description="API cost in dollars")
    latency: float = Field(ge=0, description="Response time in ms")


class HumanRewardRequest(BaseModel):
    """Human feedback on an event."""
    score: Literal[0, 1] = Field(description="Human rating: 0 (bad) or 1 (good)")
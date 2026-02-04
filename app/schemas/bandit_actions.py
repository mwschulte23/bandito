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
    score: float = Field(description="Reward value, typically [0, 1]")
    llm_output: str | Dict = Field(description="LLM response for human review")
    cost: float = Field(description="API cost in dollars")
    latency: float = Field(description="Response time in ms")


class HumanRewardRequest(BaseModel):
    """Human feedback on an event."""
    score: Literal[0, 1] = Field(description="Human rating: 0 (bad) or 1 (good)")
    


# ============ Analysis Schemas ============

class ArmLeaderboardEntry(BaseModel):
    arm_id: int
    model_name: str
    system_prompt: str
    is_active: bool
    pull_count: int
    avg_immediate_reward: float
    avg_human_reward: Optional[float] = None
    # Raw values
    avg_cost: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    total_cost: Optional[float] = None
    total_latency_ms: Optional[float] = None
    # Human-friendly display
    avg_cost_display: Optional[str] = None
    total_cost_display: Optional[str] = None
    avg_latency_display: Optional[str] = None
    total_latency_display: Optional[str] = None
    first_pull: Optional[str] = None
    last_pull: Optional[str] = None


class LeaderboardResponse(BaseModel):
    bandit_id: int
    total_pulls: int
    arms: list[ArmLeaderboardEntry]


class FeatureWeight(BaseModel):
    weight: float
    uncertainty: float
    observations: float


class ModelAnalysis(BaseModel):
    features: Dict[str, FeatureWeight]
    baseline_effect: float


class AnalysisSummary(BaseModel):
    total_features: int
    avg_observations_per_feature: float
    avg_uncertainty: float
    model_ranking: list[Dict]


class StateAnalysisResponse(BaseModel):
    bandit_id: int
    models: Dict[str, ModelAnalysis]
    prompts: Dict[str, FeatureWeight]
    summary: AnalysisSummary


# ============ Budget Schemas ============

class BudgetResponse(BaseModel):
    bandit_id: int
    budget: Optional[float]
    current_spend: float
    budget_remaining: Optional[float]
    budget_used_percent: Optional[float]
    is_over_budget: bool
from typing import Optional, Dict, Tuple
from pydantic import BaseModel


# ============ Shared Arm Bases ============

class ArmRef(BaseModel):
    """Minimal arm reference for analysis responses."""
    arm_id: int
    model_name: str
    system_prompt: str


class ArmIdentity(ArmRef):
    """Arm reference with active status."""
    is_active: bool


# ============ Reward / Leaderboard ============

class RewardBreakdown(BaseModel):
    """Transparent breakdown: raw × cost_penalty × latency_penalty = adjusted."""
    raw: Optional[float] = None
    cost_factor: Optional[float] = None
    latency_factor: Optional[float] = None
    adjusted: Optional[float] = None


class ArmLeaderboardEntry(ArmIdentity):
    # Primary metric
    human: Optional[RewardBreakdown] = None

    # Automated counterpart
    immediate: RewardBreakdown

    # Data volume
    pull_count: int
    human_count: int

    # Confidence (from A_inv posterior)
    confidence: float
    confidence_label: str

    # Supporting
    avg_cost: Optional[float] = None
    avg_latency_ms: Optional[float] = None


class LeaderboardResponse(BaseModel):
    bandit_id: int
    total_pulls: int
    total_human: int
    overall_confidence: float
    overall_confidence_label: str
    arms: list[ArmLeaderboardEntry]


# ============ State Analysis ============

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


# ============ Budget ============

class BudgetResponse(BaseModel):
    bandit_id: int
    budget: Optional[float]
    current_spend: float
    remaining: Optional[float]
    used_percent: Optional[float]
    is_over_budget: bool


# ============ Forecast ============

class ArmForecast(ArmRef):
    selection_probability: float  # P(this arm wins)
    expected_score: float         # μ = features @ theta_hat
    score_std: float              # σ = sqrt(features @ A^-1 @ features)


class ForecastResponse(BaseModel):
    bandit_id: int
    context: Dict[str, int]       # {hour_of_day, is_weekend}
    n_simulations: int
    avg_uncertainty: float        # Average score_std across arms
    confidence_note: str          # Human-readable interpretation
    arms: list[ArmForecast]


# ============ Regret ============

class RegretMetric(BaseModel):
    """Regret computation for a single metric (e.g., accuracy, adjusted)."""
    pct: float                     # % regret for this window/overall
    mean_regret: float             # average per-event gap
    mean_optimal: float            # oracle's mean reward (denominator context)
    best_arm_id: int               # the oracle arm for this metric


class RegretWindow(BaseModel):
    """One window of regret data across all metrics."""
    window: int                                          # 0-indexed
    event_range: Tuple[int, int]                         # (first_event_id, last_event_id)
    n_events: int
    metrics: Dict[str, RegretMetric]                     # "accuracy", "adjusted", etc.
    human_metrics: Dict[str, Optional[RegretMetric]]     # same keys, nullable
    human_n_events: int
    human_low_confidence: bool                           # < min_human_events


class RegretResponse(BaseModel):
    """Full regret analysis for a bandit."""
    bandit_id: int
    total_events: int
    window_size: int
    windows: list[RegretWindow]
    overall: Dict[str, RegretMetric]                     # summary across all events
    human_overall: Dict[str, Optional[RegretMetric]]


# ============ Summary ============

class ArmSummaryEntry(ArmIdentity):
    pull_count: int
    preference: Optional[int] = None  # rank by theta_hat (1 = best), None for inactive


class ArmSummaryResponse(BaseModel):
    bandit_id: int
    arms: list[ArmSummaryEntry]


class BudgetSummary(BaseModel):
    budget: Optional[float]
    current_spend: float
    used_percent: Optional[float]
    is_over_budget: bool


class ConvergenceSummary(BaseModel):
    confidence: float
    label: str


class BanditSummaryResponse(BaseModel):
    bandit_id: int
    total_pulls: int
    budget: BudgetSummary
    convergence: ConvergenceSummary

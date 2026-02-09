import io
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.models.bandit import BanditMode, BanditState


# ============ Bandit Schemas ============

class BanditCreate(BaseModel):
    name: str
    mode: BanditMode = BanditMode.experiment
    budget: Optional[float] = 2.0
    window_size: Optional[int] = 1000
    cost_importance: int = Field(default=2, ge=0, le=5, description="How much cost matters (0=ignore, 5=critical)")
    latency_importance: int = Field(default=2, ge=0, le=5, description="How much latency matters (0=ignore, 5=critical)")


class BanditRead(BaseModel):
    id: int
    user_id: int
    name: str
    mode: BanditMode
    budget: Optional[float]
    window_size: Optional[int]
    cost_importance: int
    latency_importance: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BanditUpdate(BaseModel):
    name: Optional[str] = None
    mode: Optional[BanditMode] = None
    budget: Optional[float] = None
    window_size: Optional[int] = None
    cost_importance: Optional[int] = Field(default=None, ge=0, le=5)
    latency_importance: Optional[int] = Field(default=None, ge=0, le=5)


# ============ BanditArm Schemas ============

class BanditArmCreate(BaseModel):
    # bandit_id: int
    model_name: str
    system_prompt: str
    arm_metadata: List[Dict] = []
    is_active: bool = True


class BanditArmRead(BaseModel):
    id: int
    bandit_id: int
    model_name: str
    system_prompt: str
    arm_metadata: List[Dict]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BanditArmUpdate(BaseModel):
    # DON'T ALLOW MODEL AND PROMPT UPDATE - Makes bandit "dirty"
    # model_name: Optional[str] = None
    # system_prompt: Optional[str] = None
    arm_metadata: Optional[List[Dict]] = None
    is_active: Optional[bool] = None


# ============ BanditState Schemas ============

class BanditStateRead(BaseModel):
    """API response schema for bandit state (without numpy arrays)."""
    id: int
    bandit_id: int
    dimensions: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BanditStateInternal(BaseModel):
    """Internal schema for bandit state with numpy arrays (for algorithm operations)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int
    bandit_id: int
    dimensions: int = 17
    a: np.ndarray = Field(description='dim x dim matrix')
    b: np.ndarray = Field(description='dim length vector')
    theta_hat: np.ndarray = Field(description='dim length vector')
    cholesky_l_inv: np.ndarray = Field(description='inverse cholesky decomposition')

    @classmethod
    def from_db(cls, db_state):
        """Helper to reconstruct the state from a database."""
        return cls(
            id=db_state.id,
            bandit_id=db_state.bandit_id,
            dimensions=db_state.dimensions,
            a=cls._unpack(db_state.a_bytes),
            b=cls._unpack(db_state.b_bytes),
            theta_hat=cls._unpack(db_state.theta_bytes),
            cholesky_l_inv=cls._unpack(db_state.chol_bytes)
        )

    def to_db(self) -> BanditState:
        return BanditState(
            id=self.id,
            bandit_id=self.bandit_id,
            dimensions=self.dimensions,
            a_bytes=self._pack(self.a),
            b_bytes=self._pack(self.b),
            theta_bytes=self._pack(self.theta_hat),
            chol_bytes=self._pack(self.cholesky_l_inv),
        ) 

    @staticmethod
    def _pack(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    @staticmethod
    def _unpack(blob: bytes) -> np.ndarray:
        return np.load(io.BytesIO(blob))


# ============ BanditEvent Schemas ============

class BanditEventCreate(BaseModel):
    arm_id: int
    context: Dict
    immediate_reward: float
    human_reward: Optional[float] = None
    # outcome_reward: Optional[float] = None
    cost: Optional[float] = None
    latency: Optional[float] = None


class BanditEventRead(BaseModel):
    """API response schema for bandit events."""
    id: int
    bandit_id: int
    arm_id: Optional[int]
    context: Dict
    model_score: float
    user_query: str
    llm_output: Optional[Dict] = None
    immediate_reward: Optional[float]
    human_reward: Optional[float]
    cost: Optional[float]
    latency: Optional[float]
    cost_importance: Optional[int] = None
    latency_importance: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def adjusted_immediate_reward(self) -> Optional[float]:
        if self.immediate_reward is None or self.cost_importance is None:
            return None
        from app.services.bandit.utils.rewards import calculate_reward
        return float(calculate_reward(
            self.immediate_reward, self.cost, self.latency,
            cost_importance=self.cost_importance, latency_importance=self.latency_importance
        ))

    @computed_field
    @property
    def adjusted_human_reward(self) -> Optional[float]:
        if self.human_reward is None or self.cost_importance is None:
            return None
        from app.services.bandit.utils.rewards import calculate_reward
        return float(calculate_reward(
            self.human_reward, self.cost, self.latency,
            cost_importance=self.cost_importance, latency_importance=self.latency_importance
        ))


# ============ EventSegment Schemas ============

class EventSegmentCreate(BaseModel):
    bandit_event_id: int
    segment_name: str
    segment_value: str


class EventSegmentRead(BaseModel):
    id: int
    bandit_event_id: int
    segment_name: str
    segment_value: str

    model_config = ConfigDict(from_attributes=True)


# ============ Nested/Composite Schemas ============

class BanditReadWithArms(BanditRead):
    arms: List[BanditArmRead] = []

class BanditReadArmsState(BanditRead):
    arms: List[BanditArmRead] = []
    state: BanditStateInternal


class BanditEventCreateWithSegments(BanditEventCreate):
    segments: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {'segment_name': str, 'segment_value': str}"
    )

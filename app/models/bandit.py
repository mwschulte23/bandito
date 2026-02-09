import io
# import uuid
import enum
import numpy as np
from datetime import datetime
from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship, UniqueConstraint
from sqlalchemy import Column, LargeBinary, JSON
from typing import Optional, List, Dict


class BanditMode(str, enum.Enum):
    experiment = "experiment"
    online = "online"


class Bandit(SQLModel, table=True):
    __tablename__ = "bandit"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id", ondelete="CASCADE")
    mode: BanditMode = Field(default=BanditMode.experiment)
    name: str
    budget: Optional[float] = Field(default=2)
    window_size: Optional[int] = Field(default=1000)
    cost_importance: int = Field(default=2, ge=0, le=5)
    latency_importance: int = Field(default=2, ge=0, le=5)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now, sa_column_kwargs={"onupdate": datetime.now})

    # Relationships
    arms: List["BanditArm"] = Relationship(back_populates="bandit", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    state: Optional["BanditState"] = Relationship(back_populates="bandit", sa_relationship_kwargs={"cascade": "all, delete-orphan", "uselist": False})
    events: List["BanditEvent"] = Relationship(back_populates="bandit", sa_relationship_kwargs={"cascade": "all, delete-orphan"})


class BanditArm(SQLModel, table=True):
    __tablename__ = 'bandit_arm'

    id: Optional[int] = Field(default=None, primary_key=True)
    bandit_id: int = Field(index=True, foreign_key="bandit.id", ondelete="CASCADE")

    model_name: str
    system_prompt: str
    arm_metadata: List[Dict] = Field(sa_column=Column(JSON))

    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now, sa_column_kwargs={"onupdate": datetime.now})

    # Relationships
    bandit: Optional["Bandit"] = Relationship(back_populates="arms")


class BanditState(SQLModel, table=True):
    __tablename__ = "bandit_state"

    id: Optional[int] = Field(default=None, primary_key=True)
    bandit_id: int = Field(index=True, foreign_key="bandit.id", ondelete="CASCADE")
    dimensions: int = 17
    # Storage fields (the "Blobs")
    a_bytes: bytes = Field(sa_column=Column(LargeBinary))
    b_bytes: bytes = Field(sa_column=Column(LargeBinary))
    theta_bytes: bytes = Field(sa_column=Column(LargeBinary))
    chol_bytes: bytes = Field(sa_column=Column(LargeBinary))

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now, sa_column_kwargs={"onupdate": datetime.now})
    # Relationships
    bandit: Optional["Bandit"] = Relationship(back_populates="state")

    @classmethod
    def create_new(cls, bandit_id: str, dimensions: Optional[int] = 0):
        """Initializes a new bandit state with identity matrices and zeros."""
        # TODO: Need to apply lambda on "load" to "brain" / weight inits
        A = np.eye(dimensions) # * lambda_reg # lambda_reg on init, but simplifying for now
        b = np.zeros(dimensions)
        theta_hat = np.zeros(dimensions)
        cholesky_l_inv = np.eye(dimensions)
        return cls(
            bandit_id=bandit_id,
            dimensions=dimensions,
            a_bytes=cls._pack(A),
            b_bytes=cls._pack(b),
            theta_bytes=cls._pack(theta_hat),
            chol_bytes=cls._pack(cholesky_l_inv)
        )

    def update_state(self, a, b, theta, chol):
        """Update all arrays at once from math results."""
        self.a_bytes = self._pack(a)
        self.b_bytes = self._pack(b)
        self.theta_bytes = self._pack(theta)
        self.chol_bytes = self._pack(chol)

    def resize(self, new_dimensions: int):      
        # assert new_dimensions > 0, 'new_dimensions must be > 0'
        assert new_dimensions >= self.dimensions, f'new_dimensions ({new_dimensions}) must be greater than current dimensions ({self.dimensions})'

        if self.dimensions == 0:
            self.dimensions = new_dimensions
            self.a_bytes = self._pack(np.eye(new_dimensions))
            self.b_bytes = self._pack(np.zeros(new_dimensions))
            self.theta_bytes = self._pack(np.zeros(new_dimensions))
            self.chol_bytes = self._pack(np.eye(new_dimensions))
            return
        
        A = self._unpack(self.a_bytes)
        new_A = np.eye(new_dimensions)
        new_A[:self.dimensions, :self.dimensions] = A

        b = self._unpack(self.b_bytes)
        new_b = np.zeros(new_dimensions)
        new_b[:self.dimensions] = b

        theta = self._unpack(self.theta_bytes)
        new_theta = np.zeros(new_dimensions)
        new_theta[:self.dimensions] = theta

        chol = self._unpack(self.chol_bytes)
        new_chol = np.eye(new_dimensions)
        new_chol[:self.dimensions, :self.dimensions] = chol
        self.dimensions = new_dimensions
        self.update_state(new_A, new_b, new_theta, new_chol)
    
    @staticmethod
    def _pack(arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()
    
    @staticmethod
    def _unpack(blob: bytes) -> np.ndarray:
        return np.load(io.BytesIO(blob))



class BanditEvent(SQLModel, table=True):
    __tablename__ = "bandit_event"

    id: Optional[int] = Field(default=None, primary_key=True)
    bandit_id: int = Field(index=True, foreign_key="bandit.id", ondelete="CASCADE")
    # event_id: str
    arm_id: Optional[int] = Field(default=None, foreign_key="bandit_arm.id", ondelete="CASCADE")
    context: Dict = Field(sa_column=Column(JSON))
    model_score: float
    user_query: str

    llm_output: Optional[Dict] = Field(default={}, sa_column=Column(JSON))
    immediate_reward: Optional[float] = Field(default=None)
    human_reward: Optional[float] = Field(default=None)
    # outcome_reward: Optional[float] = Field(default=None)
    cost: Optional[float] = Field(default=None)
    latency: Optional[float] = Field(default=None)
    cost_importance: Optional[int] = Field(default=None)
    latency_importance: Optional[int] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now, sa_column_kwargs={"onupdate": datetime.now})

    # Relationships
    bandit: Optional["Bandit"] = Relationship(back_populates="events")
    segments: List["EventSegment"] = Relationship(back_populates="event", sa_relationship_kwargs={"cascade": "all, delete-orphan"})


class EventSegment(SQLModel, table=True):
    '''
    A potentially KILLER feature especially in online mode.

    This is end-user context. Say mobile vs desktop. If performance diverges, we can split the bandit by the segment.

    Is the payoff to the end user worth it? We will find out!!
    '''
    __tablename__ = "bandit_segment"

    id: Optional[int] = Field(default=None, primary_key=True)
    bandit_event_id: int = Field(index=True, foreign_key="bandit_event.id", ondelete="CASCADE")

    segment_name: str
    segment_value: str

    # Relationships
    event: Optional["BanditEvent"] = Relationship(back_populates="segments")

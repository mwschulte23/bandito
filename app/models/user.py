# import uuid
from datetime import datetime
from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship, UniqueConstraint
from typing import Optional, List



class User(SQLModel, table=True):
    __tablename__ = "user"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    email: EmailStr = Field(unique=True, index=True)
    hashed_password: str
    is_superuser: bool = Field(default=False)
    # cleark_id: str
    # is_active: bool = Field(default=True)
    # is_subscribed: bool = Field(default=False)
    # subscription_id: Optional[str] = Field(default=None)
    # stripe_customer_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now, sa_column_kwargs={"onupdate": datetime.now})
    churned_at: Optional[datetime] = Field(default=None)
 
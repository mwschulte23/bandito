from sqlmodel import SQLModel
from typing import Any
from app.models.user import *
from app.models.bandit import *


metadata: Any = SQLModel.metadata
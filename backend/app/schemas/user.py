from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class UserBase(BaseModel):
    age: int
    height: float
    weight: float
    bmi: float
    marital_status: str | None = None
    family_history: bool | None = None


class UserCreate(UserBase):
    pass


class UserResponse(UserBase):
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True

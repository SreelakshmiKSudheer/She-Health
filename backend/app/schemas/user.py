from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    user_id: str # UUID from Flutter
    age: int = Field(..., gt=0, description="Age must be a positive integer")
    height: float = Field(..., gt=0, description="Height must be a positive number in cm")
    weight: float = Field(..., gt=0, description="Weight must be a positive number in kg")
    marital_status: str | None = "single"
    family_history: bool = False

class UserResponse(UserCreate):
    bmi: float
    created_at: datetime
    updated_at: datetime

class UserUpdate(BaseModel):
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    marital_status: Optional[str] = None
    family_history: Optional[bool] = None
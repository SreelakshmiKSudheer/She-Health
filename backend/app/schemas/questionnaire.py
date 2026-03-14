from pydantic import BaseModel, Field
from typing import List, Optional

class FeatureMapping(BaseModel):
    feature_name: str
    feature_value: float = 1.0

class OptionCreate(BaseModel):
    id: Optional[str] = None
    text: str
    description: Optional[str] = None  # Added here
    mappings: List[FeatureMapping]

class QuestionCreate(BaseModel):
    id: str
    text: str
    category: str
    q_type: str
    is_initial: bool = True
    priority: int = 0
    options: Optional[List[OptionCreate]] = None
    direct_mappings: Optional[List[FeatureMapping]] = None

class QuestionUpdate(BaseModel):
    text: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = None
    is_initial: Optional[bool] = None

class OptionResponse(BaseModel):
    id: str
    text: str
    description: Optional[str] = None  # Added here
    mappings: List[FeatureMapping]

class QuestionResponse(BaseModel):
    id: str
    text: str
    category: str
    q_type: str
    is_initial: bool
    priority: int
    # Change this line to allow None/Default to empty list
    options: Optional[List[OptionResponse]] = []

    class Config:
        from_attributes = True
        extra = "ignore"
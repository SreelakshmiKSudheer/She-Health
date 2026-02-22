from pydantic import BaseModel
from typing import List, Optional, Dict

class FeatureMapping(BaseModel):
    feature_name: str
    feature_value: float = 1.0

class OptionCreate(BaseModel):
    text: str
    mappings: List[FeatureMapping]

class QuestionCreate(BaseModel):
    id: str  # Unique ID like "Q_PCOS_01"
    text: str
    category: str
    q_type: str  # "yes_no", "single_select", "multi_select", "input"
    is_initial: bool = True
    priority: int
    # For single/multi select
    options: Optional[List[OptionCreate]] = None
    # For yes_no or input which represent 1 feature directly
    direct_mappings: Optional[List[FeatureMapping]] = None

# For PATCH requests
class QuestionUpdate(BaseModel):
    text: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = None
    is_initial: Optional[bool] = None

# For nested response display
class FeatureMapResponse(BaseModel):
    feature_name: str
    feature_value: float
    class Config: from_attributes = True

class OptionResponse(BaseModel):
    id: str
    text: str
    feature_mappings: List[FeatureMapResponse]
    class Config: from_attributes = True

class QuestionResponse(BaseModel):
    id: str
    text: str
    category: str
    q_type: str
    is_initial: bool
    priority: int
    options: List[OptionResponse]
    class Config: from_attributes = True
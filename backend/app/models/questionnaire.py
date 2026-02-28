from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class FeatureMappingModel(BaseModel):
    feature_name: str
    feature_value: float = 1.0

class AnswerOptionModel(BaseModel):
    id: str
    text: str
    mappings: List[FeatureMappingModel] = []

class QuestionDocument(BaseModel):
    id: str  # Custom ID like "Q_CYCLE_01"
    text: str
    category: str
    q_type: str
    is_initial: bool = True
    priority: int = 0
    options: List[AnswerOptionModel] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
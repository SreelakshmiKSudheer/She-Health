from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from datetime import datetime

class FeatureMappingModel(BaseModel):
    feature_name: str
    feature_value: float = 1.0

class AnswerOptionModel(BaseModel):
    id: str
    text: str
    description: Optional[str] = None  # New Field
    mappings: List[FeatureMappingModel] = Field(default_factory=list)

class QuestionDocument(BaseModel):
    id: str  # Custom ID like "Q_CYCLE_01"
    text: str
    category: str
    q_type: str
    is_initial: bool = True
    priority: int = 0
    options: List[AnswerOptionModel] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True)

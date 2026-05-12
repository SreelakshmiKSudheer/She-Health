from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ThyroidFeatureMapping(BaseModel):
    feature_name: str
    feature_value: float = 1.0


class ThyroidAnswerOption(BaseModel):
    id: str
    text: str
    description: Optional[str] = None
    mappings: List[ThyroidFeatureMapping] = Field(default_factory=list)


class ThyroidQuestionDocument(BaseModel):
    id: str
    text: str
    category: str
    q_type: str
    is_initial: bool = True
    priority: int = 0
    options: List[ThyroidAnswerOption] = Field(default_factory=list)
    direct_mappings: List[ThyroidFeatureMapping] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class ThyroidResponseEntry(BaseModel):
    question_id: str
    selected_option_ids: List[str]


class ThyroidResponseRecord(BaseModel):
    user_id: str
    responses: List[ThyroidResponseEntry]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ThyroidRiskDetail(BaseModel):
    probability: float
    risk_level: int
    category_name: str
    status: str
    label: Optional[str] = None
    details: Optional[str] = None
    threshold: Optional[float] = None
    predicted_positive: Optional[bool] = None


class ThyroidPredictionRecord(BaseModel):
    user_id: str
    predictions: Dict[str, ThyroidRiskDetail]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    comparison: Optional[dict] = None

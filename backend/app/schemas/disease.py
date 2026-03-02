from pydantic import BaseModel
from typing import Optional


class Disease(BaseModel):
    disease_id: str
    disease_name: str
    description: Optional[str] = None


class DiseaseQuestionMap(BaseModel):
    disease_id: str
    question_id: str
    importance: float


class Recommendation(BaseModel):
    recommendation_id: str
    disease_id: str
    risk_level: int
    diet_advice: str
    workout_advice: str
    preventive_tips: str

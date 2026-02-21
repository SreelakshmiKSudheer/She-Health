from pydantic import BaseModel
from datetime import datetime


class RiskResult(BaseModel):
    result_id: str
    user_id: str
    disease_id: str
    probability: float
    risk_level: int
    prediction_date: datetime

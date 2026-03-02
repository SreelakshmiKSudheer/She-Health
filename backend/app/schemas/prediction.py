from pydantic import BaseModel
from datetime import datetime
from typing import Dict

class RiskDetail(BaseModel):
    probability: float
    risk_level: int
    label: str

class PredictionResponse(BaseModel):
    user_id: str
    results: Dict[str, RiskDetail]
    timestamp: datetime
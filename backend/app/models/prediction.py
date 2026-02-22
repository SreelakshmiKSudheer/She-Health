from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any

class DiseaseRisk(BaseModel):
    probability: float      # e.g., 76.5
    risk_level: int         # 1-5 (as returned by your model logic)
    # We can also store the raw category name if your model provides it
    category_name: str      

class PredictionRecord(BaseModel):
    user_id: str
    predictions: Dict[str, DiseaseRisk]
    created_at: datetime = Field(default_factory=datetime.utcnow)
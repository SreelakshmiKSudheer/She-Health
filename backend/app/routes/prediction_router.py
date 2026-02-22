from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db, MongoDB
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["Predictions"])

@router.post("/{user_id}")
async def get_prediction(user_id: str, db: Session = Depends(get_db)):
    # Pass both SQL (for features) and Mongo (for results storage)
    service = PredictionService(db, MongoDB.db)
    return await service.run_full_assessment(user_id)
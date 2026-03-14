from fastapi import APIRouter, HTTPException
from app.db.database import MongoDB
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["Predictions"])

@router.post("/{user_id}")
async def get_prediction(user_id: str):
    if MongoDB.db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    
    # We pass the MongoDB database instance directly to the service
    service = PredictionService(MongoDB.db)
    return await service.run_full_assessment(user_id)


@router.get("/validate/{user_id}")
async def validate_feature_coverage(user_id: str):
    if MongoDB.db is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    service = PredictionService(MongoDB.db)
    return await service.validate_user_feature_coverage(user_id)
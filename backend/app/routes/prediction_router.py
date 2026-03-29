from fastapi import APIRouter, HTTPException
try:
    from app.db.database import MongoDB
    from app.services.prediction_service import PredictionService
except ModuleNotFoundError:
    from ..db.database import MongoDB
    from ..services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["Predictions"])

@router.post("/{user_id}")
async def get_prediction(user_id: str):
    service = PredictionService(MongoDB.db)
    return await service.run_full_assessment(user_id)


@router.get("/latest/{user_id}")
async def get_latest_prediction(user_id: str):
    service = PredictionService(MongoDB.db)
    latest = await service.get_latest_prediction(user_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="No prediction found for this user")
    return latest


@router.get("/validate/{user_id}")
async def validate_feature_coverage(user_id: str):
    service = PredictionService(MongoDB.db)
    return await service.validate_user_feature_coverage(user_id)
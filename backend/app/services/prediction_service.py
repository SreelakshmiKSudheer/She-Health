from sqlalchemy.orm import Session
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.services.response_service import ResponseService
from app.ml.predictor import SheHealthPredictor
from datetime import datetime

class PredictionService:
    def __init__(self, sql_db: Session, mongo_db: AsyncIOMotorDatabase):
        self.response_service = ResponseService(sql_db)
        self.predictor = SheHealthPredictor()
        self.collection = mongo_db.get_collection("predictions")

    async def run_assessment(self, user_id: str):
        # Fetch the feature dictionary (translated from the 31 questions)
        features = self.response_service.get_user_answers_as_feature_dict(user_id)
        
        # Get predictions directly from your specialized ML models
        results = self.predictor.predict_all_diseases(features)
        
        # Store in MongoDB for the user's history
        record = {
            "user_id": user_id,
            "predictions": results,
            "created_at": datetime.utcnow()
        }
        await self.collection.insert_one(record)
        
        return record
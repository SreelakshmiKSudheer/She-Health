from app.services.response_service import ResponseService
from app.ml.predictor import SheHealthPredictor
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

class PredictionService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.response_service = ResponseService(db)
        self.predictor = SheHealthPredictor()
        self.collection = db.get_collection("predictions")

    async def run_full_assessment(self, user_id: str):
        # 1. Get features from the Response Service (which uses MongoDB aggregation)
        features = await self.response_service.get_user_answers_as_feature_dict(user_id)      

        # 2. Get ML results
        results = self.predictor.predict_all_diseases(features)

        # 3. Store result in MongoDB
        record = {
            "user_id": user_id,
            "predictions": results,
            "created_at": datetime.utcnow()
        }
        await self.collection.insert_one(record)
        return record


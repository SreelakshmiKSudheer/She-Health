from app.services.response_service import ResponseService
from app.ml.predictor import SheHealthPredictor
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional

_NO_ID = {"_id": 0}

class PredictionService:
    _memory_predictions: dict[str, list[dict]] = {}

    def __init__(self, db: Optional[AsyncIOMotorDatabase]):
        self.response_service = ResponseService(db)
        self.predictor = SheHealthPredictor()
        self.collection = db.get_collection("predictions") if db is not None else None

    async def run_full_assessment(self, user_id: str):
        # Block prediction unless all questionnaire questions were answered.
        await self.response_service.assert_user_completed_questionnaire(user_id)

        # 1. Get features from the Response Service (which uses MongoDB aggregation)
        features = await self.response_service.get_user_answers_as_feature_dict(user_id)      

        # 2. Get ML results
        results = self.predictor.predict_all_diseases(features)

        # 3. Store result
        record = {
            "user_id": user_id,
            "predictions": results,
            "created_at": datetime.utcnow()
        }

        if self.collection is None:
            user_predictions = self._memory_predictions.setdefault(user_id, [])
            user_predictions.append(record)
            return record

        insert_result = await self.collection.insert_one(record)
        saved = await self.collection.find_one({"_id": insert_result.inserted_id}, _NO_ID)
        return saved

    async def get_latest_prediction(self, user_id: str):
        if self.collection is None:
            user_predictions = self._memory_predictions.get(user_id, [])
            if not user_predictions:
                return None
            return user_predictions[-1]

        return await self.collection.find_one(
            {"user_id": user_id},
            _NO_ID,
            sort=[("created_at", -1)],
        )

    async def validate_user_feature_coverage(self, user_id: str):
        await self.response_service.assert_user_completed_questionnaire(user_id)
        features = await self.response_service.get_user_answers_as_feature_dict(user_id)
        report = self.predictor.get_feature_coverage_report(features)
        return {
            "user_id": user_id,
            "mapped_feature_count": len(features),
            "diseases": report,
        }


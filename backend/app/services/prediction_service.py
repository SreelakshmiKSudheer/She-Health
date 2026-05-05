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

    @staticmethod
    def _risk_level_map(predictions: dict) -> dict[str, int]:
        levels: dict[str, int] = {}
        if not isinstance(predictions, dict):
            return levels

        for disease, payload in predictions.items():
            if not isinstance(payload, dict):
                continue
            try:
                levels[disease] = int(payload.get("risk_level", 0))
            except Exception:
                levels[disease] = 0
        return levels

    def _build_comparison(self, previous: Optional[dict], current_predictions: dict) -> dict:
        if not previous:
            return {
                "change_type": "first_assessment",
                "drastic_change": True,
                "changed_diseases": [],
            }

        previous_levels = self._risk_level_map(previous.get("predictions", {}))
        current_levels = self._risk_level_map(current_predictions)

        changed = []
        for disease in sorted(set(previous_levels.keys()) | set(current_levels.keys())):
            old_level = previous_levels.get(disease, 0)
            new_level = current_levels.get(disease, 0)
            if old_level != new_level:
                changed.append(
                    {
                        "disease": disease,
                        "previous_level": old_level,
                        "current_level": new_level,
                    }
                )

        drastic_change = len(changed) > 0
        return {
            "change_type": "drastic" if drastic_change else "slight",
            "drastic_change": drastic_change,
            "changed_diseases": changed,
        }

    async def _get_latest_prediction_doc(self, user_id: str):
        if self.collection is None:
            user_predictions = self._memory_predictions.get(user_id, [])
            if not user_predictions:
                return None
            return user_predictions[-1]

        return await self.collection.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)],
        )

    async def run_full_assessment(self, user_id: str):
        # Block prediction unless all questionnaire questions were answered.
        await self.response_service.assert_user_completed_questionnaire(user_id)

        # 1. Get features from the Response Service (which uses MongoDB aggregation)
        features = await self.response_service.get_user_answers_as_feature_dict(user_id)      

        # 2. Get ML results
        results = self.predictor.predict_all_diseases(features)

        previous = await self.get_latest_prediction(user_id)
        comparison = self._build_comparison(previous, results)

        # 3. Store result
        now = datetime.utcnow()
        record = {
            "user_id": user_id,
            "predictions": results,
            "created_at": now,
        }

        if self.collection is None:
            user_predictions = self._memory_predictions.setdefault(user_id, [])
            if comparison["drastic_change"] or not user_predictions:
                user_predictions.append(record)
            else:
                user_predictions[-1] = record

            return {
                **record,
                "comparison": comparison,
            }

        if comparison["drastic_change"] or previous is None:
            insert_result = await self.collection.insert_one(record)
            saved = await self.collection.find_one({"_id": insert_result.inserted_id}, _NO_ID)
            return {
                **saved,
                "comparison": comparison,
            }

        latest_doc = await self._get_latest_prediction_doc(user_id)
        if latest_doc and latest_doc.get("_id") is not None:
            await self.collection.update_one(
                {"_id": latest_doc["_id"]},
                {
                    "$set": {
                        "predictions": results,
                        "created_at": now,
                    }
                },
            )

        refreshed = await self.get_latest_prediction(user_id)
        return {
            **(refreshed or record),
            "comparison": comparison,
        }

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


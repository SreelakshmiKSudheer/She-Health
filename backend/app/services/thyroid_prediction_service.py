from __future__ import annotations

from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.ml.thyroid_predictor import ThyroidPredictor
from app.services.thyroid_service import ThyroidService

_NO_ID = {"_id": 0}


class ThyroidPredictionService:
    _memory_predictions: dict[str, list[dict]] = {}

    def __init__(self, db: Optional[AsyncIOMotorDatabase]):
        self.thyroid_service = ThyroidService(db)
        self.predictor = ThyroidPredictor()
        self.collection = db.get_collection("user_thyroid_predictions") if db is not None else None

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

    @staticmethod
    def _thyroid_snapshot(predictions: dict) -> tuple[float, int]:
        payload = predictions.get("Thyroid") if isinstance(predictions, dict) else None
        if not isinstance(payload, dict):
            return 0.0, 0

        try:
            probability = round(float(payload.get("probability", 0.0)), 2)
        except Exception:
            probability = 0.0

        try:
            risk_level = int(payload.get("risk_level", 0))
        except Exception:
            risk_level = 0

        return probability, risk_level

    def _compare_thyroid_predictions(self, previous: Optional[dict], current_predictions: dict) -> dict:
        if not previous:
            return {
                "change_type": "first_assessment",
                "drastic_change": True,
                "risk_changed": True,
                "probability_changed": True,
                "changed_diseases": [],
            }

        previous_probability, previous_risk = self._thyroid_snapshot(previous.get("predictions", {}))
        current_probability, current_risk = self._thyroid_snapshot(current_predictions)

        probability_changed = round(previous_probability, 2) != round(current_probability, 2)
        risk_changed = previous_risk != current_risk

        if risk_changed:
            change_type = "drastic"
        elif probability_changed:
            change_type = "slight"
        else:
            change_type = "no_change"

        changed = []
        if risk_changed:
            changed.append(
                {
                    "disease": "Thyroid",
                    "previous_level": previous_risk,
                    "current_level": current_risk,
                    "previous_probability": previous_probability,
                    "current_probability": current_probability,
                }
            )

        return {
            "change_type": change_type,
            "drastic_change": risk_changed,
            "risk_changed": risk_changed,
            "probability_changed": probability_changed,
            "changed_diseases": changed,
        }

    def _build_comparison(self, previous: Optional[dict], current_predictions: dict) -> dict:
        if not previous:
            return {"change_type": "first_assessment", "drastic_change": True, "changed_diseases": []}

        previous_levels = self._risk_level_map(previous.get("predictions", {}))
        current_levels = self._risk_level_map(current_predictions)

        changed = []
        for disease in sorted(set(previous_levels.keys()) | set(current_levels.keys())):
            old_level = previous_levels.get(disease, 0)
            new_level = current_levels.get(disease, 0)
            if old_level != new_level:
                changed.append({"disease": disease, "previous_level": old_level, "current_level": new_level})

        drastic_change = len(changed) > 0
        return {"change_type": "drastic" if drastic_change else "slight", "drastic_change": drastic_change, "changed_diseases": changed}

    async def _get_latest_prediction_doc(self, user_id: str):
        if self.collection is None:
            predictions = self._memory_predictions.get(user_id, [])
            if not predictions:
                return None
            return predictions[-1]

        return await self.collection.find_one({"user_id": user_id}, sort=[("created_at", -1)])

    async def get_latest_prediction(self, user_id: str):
        if self.collection is None:
            predictions = self._memory_predictions.get(user_id, [])
            if not predictions:
                return None
            return predictions[-1]

        return await self.collection.find_one({"user_id": user_id}, _NO_ID, sort=[("created_at", -1)])

    async def run_full_assessment(self, user_id: str):
        await self.thyroid_service.assert_user_completed_questionnaire(user_id)
        features = await self.thyroid_service.get_user_answers_as_feature_dict(user_id)
        results = self.predictor.predict(features)

        previous = await self.get_latest_prediction(user_id)
        comparison = self._compare_thyroid_predictions(previous, results)

        now = datetime.utcnow()
        record = {"user_id": user_id, "predictions": results, "created_at": now}

        if comparison["change_type"] == "no_change":
            return {**(previous or record), "comparison": comparison}

        if self.collection is None:
            user_predictions = self._memory_predictions.setdefault(user_id, [])
            if not user_predictions:
                user_predictions.append(record)
            elif comparison["risk_changed"]:
                user_predictions.append(record)
            else:
                user_predictions[-1] = record
            return {**record, "comparison": comparison}

        if previous is None:
            insert_result = await self.collection.insert_one(record)
            saved = await self.collection.find_one({"_id": insert_result.inserted_id}, _NO_ID)
            return {**saved, "comparison": comparison}

        if comparison["risk_changed"]:
            insert_result = await self.collection.insert_one(record)
            saved = await self.collection.find_one({"_id": insert_result.inserted_id}, _NO_ID)
            return {**saved, "comparison": comparison}

        latest_doc = await self._get_latest_prediction_doc(user_id)
        if latest_doc and latest_doc.get("_id") is not None:
            await self.collection.update_one(
                {"_id": latest_doc["_id"]},
                {"$set": {"predictions": results, "created_at": now}},
            )

        refreshed = await self.get_latest_prediction(user_id)
        return {**(refreshed or record), "comparison": comparison}

    async def validate_user_feature_coverage(self, user_id: str):
        await self.thyroid_service.assert_user_completed_questionnaire(user_id)
        features = await self.thyroid_service.get_user_answers_as_feature_dict(user_id)
        report = self.predictor.get_feature_coverage_report(features)
        return {"user_id": user_id, "mapped_feature_count": len(features), "diseases": report}

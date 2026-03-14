from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.response import SubmitResponse
from datetime import datetime

class ResponseService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.get_collection("user_responses")
        self.questions_collection = db.get_collection("questions")

    async def save_user_responses(self, data: SubmitResponse):
        # We replace the user's responses in MongoDB (Upsert)
        await self.collection.update_one(
            {"user_id": data.user_id},
            {
                "$set": {
                    "responses": [r.model_dump() for r in data.responses],
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        return {"status": "success", "message": "Responses updated in MongoDB"}

    async def get_user_answers_as_feature_dict(self, user_id: str):
        user_record = await self.collection.find_one({"user_id": user_id})
        if not user_record:
            return {}

        # Collect all selected option IDs across all questions
        selected_opt_ids = []
        for r in user_record["responses"]:
            selected_opt_ids.extend(r["selected_option_ids"])

        feature_vector = {}
        
        # Aggregate logic: Search through the nested 'options' in 'questions' collection
        pipeline = [
            {"$unwind": "$options"},
            {"$match": {"options.id": {"$in": selected_opt_ids}}},
            {"$unwind": "$options.mappings"}
        ]
        
        async for doc in self.questions_collection.aggregate(pipeline):
            mapping = doc["options"]["mappings"]
            feature_vector[mapping["feature_name"]] = mapping["feature_value"]

        return feature_vector
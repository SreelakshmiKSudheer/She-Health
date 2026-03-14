from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.questionnaire import QuestionCreate, QuestionUpdate
from fastapi import HTTPException
import uuid

# Exclude _id from every MongoDB read — single source of truth
_NO_ID = {"_id": 0}


class QuestionnaireService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.get_collection("questions")

    async def create_smart_question(self, data: QuestionCreate):
        existing = await self.collection.find_one({"id": data.id}, _NO_ID)
        if existing:
            raise HTTPException(
                status_code=400, detail=f"Question with id '{data.id}' already exists"
            )

        question_dict = data.model_dump()

        # Pop direct_mappings — used only to build the yes_no option, must not be persisted
        direct_mappings = question_dict.pop("direct_mappings", None)

        if question_dict.get("options") is None:
            question_dict["options"] = []

        # Auto-generate the Yes option for yes_no questions
        if data.q_type == "yes_no" and direct_mappings:
            question_dict["options"] = [
                {
                    "id": f"OPT_{data.id}_YES",
                    "text": "Yes",
                    "description": "Select if this applies to you",
                    "mappings": direct_mappings,
                }
            ]

        # Auto-assign IDs to options that were submitted without one
        for opt in question_dict["options"]:
            if not opt.get("id"):
                opt["id"] = f"OPT_{uuid.uuid4().hex[:8].upper()}"

        await self.collection.insert_one(question_dict)
        question_dict.pop("_id", None)  # Motor adds _id in-place after insert
        return question_dict

    async def list_all_questions(self):
        cursor = self.collection.find({}, _NO_ID).sort("priority", 1)
        questions = await cursor.to_list(length=200)
        for q in questions:
            if q.get("options") is None:
                q["options"] = []
        return questions

    async def get_question(self, q_id: str):
        q = await self.collection.find_one({"id": q_id}, _NO_ID)
        if q and q.get("options") is None:
            q["options"] = []
        return q

    async def update_question(self, q_id: str, data: QuestionUpdate):
        update_data = data.model_dump(exclude_unset=True)
        result = await self.collection.update_one({"id": q_id}, {"$set": update_data})
        if result.matched_count == 0:
            return None
        return await self.get_question(q_id)

    async def delete_question(self, q_id: str):
        result = await self.collection.delete_one({"id": q_id})
        return result.deleted_count > 0
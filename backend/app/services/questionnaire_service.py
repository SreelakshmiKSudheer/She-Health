from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.questionnaire import QuestionCreate, QuestionUpdate
import uuid

class QuestionnaireService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.get_collection("questions")

    async def create_smart_question(self, data: QuestionCreate):
        question_dict = data.model_dump()
        
        # 1. Automation for Yes/No
        if data.q_type == "yes_no" and data.direct_mappings:
            question_dict["options"] = [{
                "id": f"OPT_{data.id}_YES",
                "text": "Yes",
                "mappings": [m.model_dump() for m in data.direct_mappings]
            }]
        
        # 2. Ensure all options have IDs
        if question_dict.get("options"):
            for opt in question_dict["options"]:
                if not opt.get("id"):
                    opt["id"] = f"OPT_{uuid.uuid4().hex[:8].upper()}"

        # 3. Save to MongoDB
        await self.collection.insert_one(question_dict)
        
        # 4. Remove _id to prevent FastAPI serialization error
        if "_id" in question_dict:
            del question_dict["_id"]
            
        return question_dict

    async def list_all_questions(self):
        cursor = self.collection.find().sort("priority", 1)
        questions = await cursor.to_list(length=100)
        # Clean the list for JSON serialization
        for q in questions:
            if "_id" in q:
                del q["_id"]
        return questions

    async def get_question(self, q_id: str):
        q = await self.collection.find_one({"id": q_id})
        if q and "_id" in q:
            del q["_id"]
        return q

    async def update_question(self, q_id: str, data: QuestionUpdate):
        update_data = data.model_dump(exclude_unset=True)
        await self.collection.update_one({"id": q_id}, {"$set": update_data})
        return await self.get_question(q_id)

    async def delete_question(self, q_id: str):
        result = await self.collection.delete_one({"id": q_id})
        return result.deleted_count > 0
from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
from app.models.user import UserProfile
from app.schemas.user import UserCreate

_NO_ID = {"_id": 0}

class UserService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db.get_collection("users")

    async def create_user_profile(self, user_data: UserCreate):
        profile = UserProfile(**user_data.model_dump())
        await self.collection.insert_one(profile.model_dump())
        return profile

    async def get_user_by_id(self, user_id: str):
        return await self.collection.find_one({"user_id": user_id}, _NO_ID)

    async def list_all_users(self):
        # Returns a list of all user documents
        cursor = self.collection.find({}, _NO_ID)
        return await cursor.to_list(length=100) # Adjust length as needed

    async def update_user_profile(self, user_id: str, update_data: dict):
        # Update timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        # Recalculate BMI if height or weight changes
        if "height" in update_data or "weight" in update_data:
            current = await self.get_user_by_id(user_id)
            if current:
                h = update_data.get("height", current.get("height"))
                w = update_data.get("weight", current.get("weight"))
                update_data["bmi"] = round(w / ((h / 100) ** 2), 2)

        result = await self.collection.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    async def delete_user_profile(self, user_id: str):
        result = await self.collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0
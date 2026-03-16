from motor.motor_asyncio import AsyncIOMotorDatabase
from datetime import datetime
from app.models.user import UserProfile
from app.schemas.user import UserCreate
from typing import Optional

_NO_ID = {"_id": 0}

class UserService:
    _memory_users: dict[str, dict] = {}

    def __init__(self, db: Optional[AsyncIOMotorDatabase]):
        self.collection = db.get_collection("users") if db is not None else None

    async def create_user_profile(self, user_data: UserCreate):
        profile = UserProfile(**user_data.model_dump())
        if self.collection is None:
            self._memory_users[profile.user_id] = profile.model_dump()
            return profile

        await self.collection.insert_one(profile.model_dump())
        return profile

    async def get_user_by_id(self, user_id: str):
        if self.collection is None:
            return self._memory_users.get(user_id)
        return await self.collection.find_one({"user_id": user_id}, _NO_ID)

    async def list_all_users(self):
        if self.collection is None:
            return list(self._memory_users.values())

        # Returns a list of all user documents
        cursor = self.collection.find({}, _NO_ID)
        return await cursor.to_list(length=100) # Adjust length as needed

    async def update_user_profile(self, user_id: str, update_data: dict):
        if self.collection is None:
            current = self._memory_users.get(user_id)
            if not current:
                return False

            update_data["updated_at"] = datetime.utcnow()
            if "height" in update_data or "weight" in update_data:
                h = update_data.get("height", current.get("height"))
                w = update_data.get("weight", current.get("weight"))
                update_data["bmi"] = round(w / ((h / 100) ** 2), 2)

            self._memory_users[user_id] = {**current, **update_data}
            return True

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
        if self.collection is None:
            return self._memory_users.pop(user_id, None) is not None

        result = await self.collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0
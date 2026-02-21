from datetime import datetime
from app.config.database import get_database
import uuid


async def create_user(user_data: dict):
    db = get_database()
    user_data["user_id"] = str(uuid.uuid4())
    user_data["created_at"] = datetime.utcnow()

    await db.users.insert_one(user_data)
    return user_data


async def get_user(user_id: str):
    db = get_database()
    return await db.users.find_one({"user_id": user_id})

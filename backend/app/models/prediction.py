from app.config.database import get_database
from datetime import datetime
import uuid


async def save_risk_result(result_data: dict):
    db = get_database()
    result_data["result_id"] = str(uuid.uuid4())
    result_data["prediction_date"] = datetime.utcnow()

    await db.risk_results.insert_one(result_data)
    return result_data


async def get_user_results(user_id: str):
    db = get_database()
    return await db.risk_results.find({"user_id": user_id}).to_list(100)

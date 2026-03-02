from app.config.database import get_database
import uuid


async def create_disease(disease_data: dict):
    db = get_database()
    disease_data["disease_id"] = str(uuid.uuid4())

    await db.diseases.insert_one(disease_data)
    return disease_data


async def create_recommendation(data: dict):
    db = get_database()
    data["recommendation_id"] = str(uuid.uuid4())

    await db.recommendations.insert_one(data)
    return data

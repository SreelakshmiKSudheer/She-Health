from app.db.database import MongoDB
import uuid


async def create_disease(disease_data: dict):
    db = MongoDB.db
    disease_data["disease_id"] = str(uuid.uuid4())

    await db.diseases.insert_one(disease_data)
    return disease_data


async def create_recommendation(data: dict):
    db = MongoDB.db
    data["recommendation_id"] = str(uuid.uuid4())

    await db.recommendations.insert_one(data)
    return data

import os
import motor.motor_asyncio
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect_to_mongo(cls):
        mongodb_url = os.getenv("MONGODB_URL")
        db_name = os.getenv("DB_NAME", "She_Health")
        cls.client = motor.motor_asyncio.AsyncIOMotorClient(
            mongodb_url,
            server_api=ServerApi(version="1", strict=True, deprecation_errors=True)     
        )
        cls.db = cls.client[db_name]
        print(f"✅ Connected to MongoDB: {db_name}")

    @classmethod
    async def close_mongo_connection(cls):
        if cls.client:
            cls.client.close()
            print("❌ MongoDB connection closed.")
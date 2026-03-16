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
        if not mongodb_url:
            cls.client = None
            cls.db = None
            print("⚠️ MONGODB_URL is not configured. Backend is running in offline mode.")
            return

        try:
            cls.client = motor.motor_asyncio.AsyncIOMotorClient(
                mongodb_url,
                server_api=ServerApi(version="1", strict=True, deprecation_errors=True),
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
            )

            # Force an initial server selection so startup fails fast if DB is unreachable.
            await cls.client.admin.command("ping")
            cls.db = cls.client[db_name]
            print(f"✅ Connected to MongoDB: {db_name}")
        except Exception as exc:
            cls.client = None
            cls.db = None
            print(f"⚠️ MongoDB unavailable ({exc}). Backend is running in offline mode.")

    @classmethod
    async def close_mongo_connection(cls):
        if cls.client:
            cls.client.close()
            print("❌ MongoDB connection closed.")
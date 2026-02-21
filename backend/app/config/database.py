from pymongo import AsyncMongoClient
from app.config.settings import settings

# MongoDB connection
conn: AsyncMongoClient = None
db = None

async def connect_to_mongo():
    global conn, db
    conn = AsyncMongoClient(settings.mongodb_url)
    db = conn[settings.database_name]
    print("✓ Connected to MongoDB")

async def close_mongo_connection():
    global conn
    if conn:
        conn.close()
        print("✓ Disconnected from MongoDB")

async def get_database():
    return db

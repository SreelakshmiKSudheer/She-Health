import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.base import Base
from dotenv import load_dotenv
import motor.motor_asyncio
from pymongo.server_api import ServerApi

load_dotenv()

# --- SQL CONFIG (SQLite) ---
SQLALCHEMY_DATABASE_URL = os.getenv("SQL_DATABASE_URL", "sqlite:///./shehealth_server.db")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- MONGODB CONFIG ---
class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect_to_mongo(cls):
        mongodb_url = os.getenv("MONGODB_URL")
        db_name = os.getenv("DB_NAME", "shehealth_db")
        cls.client = motor.motor_asyncio.AsyncIOMotorClient(
            mongodb_url,
            server_api=ServerApi(version="1", strict=True, deprecation_errors=True)
        )
        cls.db = cls.client[db_name]

    @classmethod
    async def close_mongo_connection(cls):
        if cls.client:
            cls.client.close()

# --- INITIALIZATION FUNCTION ---
def init_db():
    """This is the function main.py is looking for."""
    # We import models here to register them with Base.metadata
    import app.models.questionnaire 
    Base.metadata.create_all(bind=engine)
    print("✅ SQL Database and Tables initialized.")
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.database import MongoDB, init_db # Ensure this line is correct
from app.routes import user_router, questionnaire_router, response_router, prediction_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Connect to MongoDB
    await MongoDB.connect_to_mongo()
    # 2. Startup: Initialize SQLite Tables
    init_db()
    yield
    # 3. Shutdown
    await MongoDB.close_mongo_connection()

app = FastAPI(lifespan=lifespan)
app.include_router(user_router.router)
app.include_router(questionnaire_router.router)
app.include_router(response_router.router)
app.include_router(prediction_router.router)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.database import MongoDB
from app.routes import user_router, questionnaire_router, response_router, prediction_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    await MongoDB.connect_to_mongo()
    yield
    # Shutdown: Close connection
    await MongoDB.close_mongo_connection()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router.router)
app.include_router(questionnaire_router.router)
app.include_router(response_router.router)
app.include_router(prediction_router.router)
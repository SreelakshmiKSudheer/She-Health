import os
from dotenv import load_dotenv
import sys

app_path = os.path.dirname(os.path.dirname(__file__))
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.db.database import MongoDB
    from app.routes import (
        user_router,
        questionnaire_router,
        response_router,
        prediction_router,
        cycle_router,
        cycle_prediction_router,
    )
except ModuleNotFoundError:
    from .db.database import MongoDB
    from .routes import (
        user_router,
        questionnaire_router,
        response_router,
        prediction_router,
        cycle_router,
        cycle_prediction_router,
    )

load_dotenv()
print(f"DEBUG: MONGODB_URL is {os.getenv('MONGODB_URL')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await MongoDB.connect_to_mongo()
    yield
    await MongoDB.close_mongo_connection()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(user_router.router,
    prefix="/users", tags=["Users"])

app.include_router(questionnaire_router.router)
app.include_router(response_router.router)

app.include_router(prediction_router.router,
    prefix="/api/predict", tags=["Disease Prediction"])

# Cycle logging & history  →  /api/cycle/...
#   POST   /api/cycle/log
#   GET    /api/cycle/history
#   DELETE /api/cycle/{id}
app.include_router(cycle_router.router,
    prefix="/api/cycle", tags=["Cycle"])

# ML cycle prediction  →  /api/cycle-predict/...
#   POST  /api/cycle-predict/predict
#   POST  /api/cycle-predict/bulk
app.include_router(cycle_prediction_router.router,
    prefix="/api/cycle-predict", tags=["Cycle Prediction"])
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from app.db.database import MongoDB
from app.models.thyroid import ThyroidQuestionDocument
from app.schemas.response import PartialSubmitResponse, SubmitResponse
from app.services.thyroid_prediction_service import ThyroidPredictionService
from app.services.thyroid_service import ThyroidService

router = APIRouter(prefix="/thyroid", tags=["Thyroid"])


def _get_service() -> ThyroidService:
    return ThyroidService(MongoDB.db)


def _get_prediction_service() -> ThyroidPredictionService:
    return ThyroidPredictionService(MongoDB.db)


@router.get("/questions", response_model=List[ThyroidQuestionDocument])
async def get_thyroid_questions():
    return await _get_service().list_questions()


@router.post("/responses/submit")
async def submit_thyroid_responses(data: SubmitResponse):
    return await _get_service().save_user_responses(data)


@router.patch("/responses/update")
async def update_thyroid_responses(data: PartialSubmitResponse):
    return await _get_service().update_user_responses(data)


@router.get("/responses/{user_id}")
async def get_thyroid_responses(user_id: str):
    return await _get_service().get_user_responses(user_id)


@router.post("/predict/{user_id}")
async def run_thyroid_prediction(user_id: str):
    return await _get_prediction_service().run_full_assessment(user_id)


@router.get("/predict/latest/{user_id}")
async def get_latest_thyroid_prediction(user_id: str):
    latest = await _get_prediction_service().get_latest_prediction(user_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="No thyroid prediction found for this user")
    return latest


@router.get("/predict/validate/{user_id}")
async def validate_thyroid_feature_coverage(user_id: str):
    return await _get_prediction_service().validate_user_feature_coverage(user_id)

from fastapi import APIRouter, HTTPException
from app.schemas.response import SubmitResponse, UserResponsesOut
from app.services.response_service import ResponseService
from app.db.database import MongoDB

router = APIRouter(prefix="/response", tags=["Response"])


def _get_service() -> ResponseService:
    if MongoDB.db is None:
        raise HTTPException(status_code=500, detail="Database not connected")
    return ResponseService(MongoDB.db)


@router.post("/submit")
async def submit_user_responses(data: SubmitResponse):
    return await _get_service().save_user_responses(data)


@router.get("/{user_id}", response_model=UserResponsesOut)
async def get_user_responses(user_id: str):
    return await _get_service().get_user_responses(user_id)


from app.schemas.response import SubmitResponse
from app.services.response_service import ResponseService
from fastapi import APIRouter, Depends
from app.db.database import MongoDB

router = APIRouter(prefix="/response", tags=["Response"])

@router.post("/submit")
async def submit_user_responses(data: SubmitResponse):
    service = ResponseService(MongoDB.db)
    return await service.save_user_responses(data)


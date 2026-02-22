from app.schemas.response import SubmitResponse
from app.services.response_service import ResponseService
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db

router = APIRouter(prefix="/response", tags=["Response"])

@router.post("/submit")
async def submit_user_responses(data: SubmitResponse, db: Session = Depends(get_db)):
    service = ResponseService(db)
    return await service.save_user_responses(data)
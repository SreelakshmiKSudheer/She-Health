from fastapi import APIRouter, HTTPException
from typing import List
from app.db.database import MongoDB
from app.services.questionnaire_service import QuestionnaireService
from app.schemas.questionnaire import QuestionCreate, QuestionUpdate, QuestionResponse

router = APIRouter(prefix="/questionnaire", tags=["Questionnaire"])

@router.post("/create", response_model=QuestionResponse)
async def create_question(data: QuestionCreate):
    service = QuestionnaireService(MongoDB.db)
    return await service.create_smart_question(data)

@router.get("/", response_model=List[QuestionResponse])
async def list_questions():
    service = QuestionnaireService(MongoDB.db)
    return await service.list_all_questions()

@router.get("/{q_id}", response_model=QuestionResponse)
async def get_question(q_id: str):
    service = QuestionnaireService(MongoDB.db)
    question = await service.get_question(q_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question

@router.patch("/{q_id}", response_model=QuestionResponse)
async def update_question(q_id: str, data: QuestionUpdate):
    service = QuestionnaireService(MongoDB.db)
    updated = await service.update_question(q_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Question not found")
    return updated

@router.delete("/{q_id}")
async def remove_question(q_id: str):
    service = QuestionnaireService(MongoDB.db)
    if not await service.delete_question(q_id):
        raise HTTPException(status_code=404, detail="Question not found")
    return {"message": "Question and associated data deleted from MongoDB"}
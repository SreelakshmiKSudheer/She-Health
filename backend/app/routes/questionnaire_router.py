from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.services.questionnaire_service import QuestionnaireService
from app.schemas.questionnaire import QuestionCreate, QuestionUpdate, QuestionResponse
from typing import List

router = APIRouter(prefix="/questionnaire", tags=["Questionnaire"])

@router.post("/create")
def create_question(data: QuestionCreate, db: Session = Depends(get_db)):
    service = QuestionnaireService(db)
    return service.create_smart_question(data)

@router.get("/")
def list_questions(db: Session = Depends(get_db)):
    service = QuestionnaireService(db)
    return service.get_all_questions()

@router.delete("/{q_id}")
def remove_question(q_id: str, db: Session = Depends(get_db)):
    service = QuestionnaireService(db)
    if not service.delete_question(q_id):
        raise HTTPException(status_code=404, detail="Question not found")
    return {"message": "Question and all associated mappings deleted"}

@router.get("/", response_model=List[QuestionResponse])
def get_all_questions(db: Session = Depends(get_db)):
    service = QuestionnaireService(db)
    return service.list_all_questions()

@router.patch("/{q_id}", response_model=QuestionResponse)
def update_question(q_id: str, data: QuestionUpdate, db: Session = Depends(get_db)):
    service = QuestionnaireService(db)
    updated = service.update_question(q_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Question not found")
    return updated
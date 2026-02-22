from sqlalchemy.orm import Session, joinedload
from app.models.questionnaire import Question, AnswerOption, OptionFeatureMap
from app.schemas.questionnaire import QuestionCreate, QuestionUpdate
import uuid

class QuestionnaireService:
    def __init__(self, db: Session):
        self.db = db

    def create_smart_question(self, data: QuestionCreate):
        # 1. Create the Base Question
        new_q = Question(
            id=data.id,
            text=data.text,
            category=data.category,
            q_type=data.q_type,
            is_initial=data.is_initial,
            priority=data.priority
        )
        self.db.add(new_q)

        # 2. Automation for Yes/No
        if data.q_type == "yes_no":
            # Automatically create 'Yes' option linked to the feature
            yes_opt = AnswerOption(id=f"OPT_{data.id}_YES", question_id=data.id, text="Yes")
            self.db.add(yes_opt)
            self.db.flush() # Get IDs
            
            for mapping in data.direct_mappings:
                f_map = OptionFeatureMap(
                    option_id=yes_opt.id,
                    feature_name=mapping.feature_name,
                    feature_value=mapping.feature_value
                )
                self.db.add(f_map)

        # 3. Handle Multi/Single Select
        elif data.q_type in ["single_select", "multi_select"] and data.options:
            for opt_data in data.options:
                opt_id = f"OPT_{uuid.uuid4().hex[:8].upper()}"
                new_opt = AnswerOption(id=opt_id, question_id=data.id, text=opt_data.text)
                self.db.add(new_opt)
                self.db.flush()

                for mapping in opt_data.mappings:
                    f_map = OptionFeatureMap(
                        option_id=new_opt.id,
                        feature_name=mapping.feature_name,
                        feature_value=mapping.feature_value
                    )
                    self.db.add(f_map)

        self.db.commit()
        return new_q

    def get_all_questions(self):
        # Joins everything to return a complete nested object
        return self.db.query(Question).order_by(Question.priority).all()

    def get_question(self, q_id: str):
        return self.db.query(Question).filter(Question.id == q_id).first()

    def delete_question(self, q_id: str):
        # Because we used cascade="all, delete-orphan" in our Model definition,
        # deleting the Question automatically purges Options and FeatureMaps.
        target = self.get_question(q_id)
        if target:
            self.db.delete(target)
            self.db.commit()
            return True
        return False
    
    def list_all_questions(self):
        """
        Fetches all questions with their nested options and mappings.
        We use joinedload to prevent 'N+1' query issues and get everything in one go.
        """
        return self.db.query(Question).options(
            joinedload(Question.options).joinedload(AnswerOption.feature_mappings)
        ).order_by(Question.priority).all()

    def update_question(self, q_id: str, data: QuestionUpdate):
        """
        Updates basic question details. 
        Note: For structural changes (like changing options/mappings), 
        it is usually safer to delete and recreate, but this handles basic text/priority.
        """
        db_question = self.get_question(q_id)
        if not db_question:
            return None

        # Update base fields
        update_data = data.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_question, key, value)

        self.db.commit()
        self.db.refresh(db_question)
        return db_question
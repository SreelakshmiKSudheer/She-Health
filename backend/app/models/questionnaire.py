from sqlalchemy import Column, String, Integer, Boolean, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.db.base import Base # This should be your SQLAlchemy declarative_base
from datetime import datetime

class Question(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True, index=True) # e.g., "Q_SYMPTOMS_01"
    text = Column(String, nullable=False)
    category = Column(String)             # e.g., "PCOS", "General", "Thyroid"
    q_type = Column(String, nullable=False) # "yes_no", "single_select", "multi_select", "input"
    is_initial = Column(Boolean, default=True)
    priority = Column(Integer, default=0)

    # Relationship to AnswerOptions with Cascade Delete
    options = relationship(
        "AnswerOption", 
        back_populates="question", 
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class AnswerOption(Base):
    __tablename__ = "answer_options"

    id = Column(String, primary_key=True, index=True)
    question_id = Column(String, ForeignKey("questions.id", ondelete="CASCADE"))
    text = Column(String, nullable=False) # The label user sees (e.g., "Yes", "Irregular periods")

    question = relationship("Question", back_populates="options")
    
    # Relationship to Feature Maps with Cascade Delete
    feature_mappings = relationship(
        "OptionFeatureMap", 
        back_populates="option", 
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class OptionFeatureMap(Base):
    """
    The 'Intelligence Layer' that maps a single UI choice 
    to one or more of the 125 ML features.
    """
    __tablename__ = "option_feature_maps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    option_id = Column(String, ForeignKey("answer_options.id", ondelete="CASCADE"))
    feature_name = Column(String, nullable=False) # e.g., "pcos_acne_binary"
    feature_value = Column(Float, default=1.0)    # The value fed to the ML model

    option = relationship("AnswerOption", back_populates="feature_mappings")

class UserResponse(Base):
    __tablename__ = "user_responses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True) # The UUID from Flutter/Mongo
    question_id = Column(String, ForeignKey("questions.id"))
    option_id = Column(String, ForeignKey("answer_options.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
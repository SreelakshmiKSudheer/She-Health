from pydantic import BaseModel
from typing import List
from datetime import datetime


# Question
class QuestionBase(BaseModel):
    question_text: str
    category: str
    question_type: str
    is_initial: bool
    priority: int


class QuestionResponse(QuestionBase):
    question_id: str


# Answer Option
class AnswerOption(BaseModel):
    option_id: str
    question_id: str
    option_text: str


# Option → Feature Mapping
class OptionFeatureMap(BaseModel):
    option_id: str
    feature_name: str
    feature_value: float


# User Response
class UserResponseCreate(BaseModel):
    user_id: str
    question_id: str
    selected_options: List[str]


class UserResponseOut(UserResponseCreate):
    response_id: str
    timestamp: datetime

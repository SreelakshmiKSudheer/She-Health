from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime


class SingleResponse(BaseModel):
    question_id: str
    selected_option_ids: List[str]

    @field_validator("selected_option_ids")
    @classmethod
    def must_not_be_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("selected_option_ids must contain at least one value")
        return v


class SubmitResponse(BaseModel):
    user_id: str
    responses: List[SingleResponse]

    @field_validator("responses")
    @classmethod
    def responses_must_not_be_empty(cls, v: List[SingleResponse]) -> List[SingleResponse]:
        if not v:
            raise ValueError("responses list must not be empty")
        return v


class SingleResponseOut(BaseModel):
    question_id: str
    selected_option_ids: List[str]


class UserResponsesOut(BaseModel):
    user_id: str
    responses: List[SingleResponseOut]
    updated_at: Optional[datetime] = None
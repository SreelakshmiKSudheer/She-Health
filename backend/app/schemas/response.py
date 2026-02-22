from pydantic import BaseModel
from typing import List

class SingleResponse(BaseModel):
    question_id: str
    selected_option_ids: List[str] # List handles Multi-Select and Single-Select (length 1)

class SubmitResponse(BaseModel):
    user_id: str
    responses: List[SingleResponse]
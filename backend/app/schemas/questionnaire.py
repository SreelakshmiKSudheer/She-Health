from pydantic import BaseModel, ConfigDict, model_validator
from typing import List, Literal, Optional


class FeatureMapping(BaseModel):
    feature_name: str
    feature_value: float = 1.0


class OptionCreate(BaseModel):
    id: Optional[str] = None
    text: str
    description: Optional[str] = None
    mappings: List[FeatureMapping]


class QuestionCreate(BaseModel):
    id: str
    text: str
    category: str
    q_type: Literal["yes_no", "single_select", "multi_select", "input"]
    is_initial: bool = True
    priority: int = 0
    options: Optional[List[OptionCreate]] = None
    direct_mappings: Optional[List[FeatureMapping]] = None

    @model_validator(mode="after")
    def validate_q_type_shape(self) -> "QuestionCreate":
        qt = self.q_type
        has_options = bool(self.options)
        has_direct = bool(self.direct_mappings)

        if qt == "yes_no":
            if not has_direct:
                raise ValueError("yes_no questions require direct_mappings")
            if has_options:
                raise ValueError(
                    "yes_no questions must not include options; use direct_mappings instead"
                )
        elif qt in ("single_select", "multi_select"):
            if not has_options:
                raise ValueError(f"{qt} questions require at least one option")
            if has_direct:
                raise ValueError(
                    f"{qt} questions must not use direct_mappings; embed mappings in each option"
                )
        elif qt == "input":
            if has_options:
                raise ValueError("input questions must not include options")
        return self


class QuestionUpdate(BaseModel):
    text: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = None
    is_initial: Optional[bool] = None


class OptionResponse(BaseModel):
    id: str
    text: str
    description: Optional[str] = None
    mappings: List[FeatureMapping]


class QuestionResponse(BaseModel):
    id: str
    text: str
    category: str
    q_type: str
    is_initial: bool
    priority: int
    options: Optional[List[OptionResponse]] = []

    model_config = ConfigDict(from_attributes=True, extra="ignore")
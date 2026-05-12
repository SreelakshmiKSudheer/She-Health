from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.response import PartialSubmitResponse, SubmitResponse

_NO_ID = {"_id": 0}

_OFFLINE_THYROID_QUESTIONS = [
    {
        "id": "Q_THYROID_AGE",
        "text": "What is your age?",
        "category": "thyroid",
        "q_type": "input",
        "is_initial": True,
        "priority": 1,
        "options": [],
        "direct_mappings": [{"feature_name": "age", "feature_value": 1.0}],
    },
    {
        "id": "Q_THYROID_SEX",
        "text": "Select your sex assigned at birth.",
        "category": "thyroid",
        "q_type": "single_select",
        "is_initial": True,
        "priority": 2,
        "options": [
            {
                "id": "OPT_Q_THYROID_SEX_01",
                "text": "Female",
                "description": None,
                "mappings": [{"feature_name": "sex", "feature_value": 1.0}],
            },
            {
                "id": "OPT_Q_THYROID_SEX_02",
                "text": "Male",
                "description": None,
                "mappings": [{"feature_name": "sex", "feature_value": 0.0}],
            },
        ],
    },
    {
        "id": "Q_THYROID_SYMPTOMS",
        "text": "Which thyroid-related history items apply to you?",
        "category": "thyroid",
        "q_type": "multi_select",
        "is_initial": True,
        "priority": 3,
        "options": [
            {"id": "OPT_Q_THYROID_SYMPTOMS_01", "text": "On thyroxine", "description": None, "mappings": [{"feature_name": "on thyroxine", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_02", "text": "Query on thyroxine", "description": None, "mappings": [{"feature_name": "query on thyroxine", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_03", "text": "On antithyroid medication", "description": None, "mappings": [{"feature_name": "on antithyroid medication", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_04", "text": "Thyroid surgery", "description": None, "mappings": [{"feature_name": "thyroid surgery", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_05", "text": "I131 treatment", "description": None, "mappings": [{"feature_name": "I131 treatment", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_06", "text": "Query hypothyroid", "description": None, "mappings": [{"feature_name": "query hypothyroid", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_07", "text": "Query hyperthyroid", "description": None, "mappings": [{"feature_name": "query hyperthyroid", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_08", "text": "Lithium use", "description": None, "mappings": [{"feature_name": "lithium", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_09", "text": "Goitre", "description": None, "mappings": [{"feature_name": "goitre", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_10", "text": "Tumor", "description": None, "mappings": [{"feature_name": "tumor", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_11", "text": "Hypopituitary", "description": None, "mappings": [{"feature_name": "hypopituitary", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_12", "text": "Psychological treatment", "description": None, "mappings": [{"feature_name": "psych", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_13", "text": "Recent illness", "description": None, "mappings": [{"feature_name": "sick", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_SYMPTOMS_14", "text": "Pregnant", "description": None, "mappings": [{"feature_name": "pregnant", "feature_value": 1.0}]},
        ],
    },
    {
        "id": "Q_THYROID_LAB_FLAGS",
        "text": "Which thyroid lab tests have been measured recently?",
        "category": "thyroid",
        "q_type": "multi_select",
        "is_initial": True,
        "priority": 4,
        "options": [
            {"id": "OPT_Q_THYROID_LAB_FLAGS_01", "text": "TSH measured", "description": None, "mappings": [{"feature_name": "TSH measured", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_LAB_FLAGS_02", "text": "T3 measured", "description": None, "mappings": [{"feature_name": "T3 measured", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_LAB_FLAGS_03", "text": "TT4 measured", "description": None, "mappings": [{"feature_name": "TT4 measured", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_LAB_FLAGS_04", "text": "T4U measured", "description": None, "mappings": [{"feature_name": "T4U measured", "feature_value": 1.0}]},
            {"id": "OPT_Q_THYROID_LAB_FLAGS_05", "text": "FTI measured", "description": None, "mappings": [{"feature_name": "FTI measured", "feature_value": 1.0}]},
        ],
    },
    {
        "id": "Q_THYROID_TSH",
        "text": "What is your TSH value?",
        "category": "thyroid",
        "q_type": "input",
        "is_initial": True,
        "priority": 5,
        "options": [],
        "direct_mappings": [{"feature_name": "TSH", "feature_value": 1.0}],
    },
    {
        "id": "Q_THYROID_T3",
        "text": "What is your T3 value?",
        "category": "thyroid",
        "q_type": "input",
        "is_initial": True,
        "priority": 6,
        "options": [],
        "direct_mappings": [{"feature_name": "T3", "feature_value": 1.0}],
    },
    {
        "id": "Q_THYROID_TT4",
        "text": "What is your TT4 value?",
        "category": "thyroid",
        "q_type": "input",
        "is_initial": True,
        "priority": 7,
        "options": [],
        "direct_mappings": [{"feature_name": "TT4", "feature_value": 1.0}],
    },
    {
        "id": "Q_THYROID_T4U",
        "text": "What is your T4U value?",
        "category": "thyroid",
        "q_type": "input",
        "is_initial": True,
        "priority": 8,
        "options": [],
        "direct_mappings": [{"feature_name": "T4U", "feature_value": 1.0}],
    },
    {
        "id": "Q_THYROID_FTI",
        "text": "What is your FTI value?",
        "category": "thyroid",
        "q_type": "input",
        "is_initial": True,
        "priority": 9,
        "options": [],
        "direct_mappings": [{"feature_name": "FTI", "feature_value": 1.0}],
    },
]


class ThyroidService:
    _memory_questions: list[dict] | None = None
    _memory_responses: dict[str, dict] = {}

    def __init__(self, db: Optional[AsyncIOMotorDatabase]):
        self.questions_collection = db.get_collection("thyroid_questions") if db is not None else None
        self.responses_collection = db.get_collection("user_thyroid_responses") if db is not None else None

    @staticmethod
    def _normalize_question(question: dict) -> dict:
        normalized = dict(question)
        normalized.setdefault("options", [])
        normalized.setdefault("direct_mappings", [])
        return normalized

    async def _get_question_docs(self) -> list[dict]:
        if self.questions_collection is None:
            if self._memory_questions is None:
                self._memory_questions = deepcopy(_OFFLINE_THYROID_QUESTIONS)
            return deepcopy(self._memory_questions)

        try:
            cursor = self.questions_collection.find({}, _NO_ID).sort("priority", 1)
            docs = [self._normalize_question(doc) async for doc in cursor]
            if not docs:
                return deepcopy(_OFFLINE_THYROID_QUESTIONS)
            return docs
        except Exception:
            return deepcopy(_OFFLINE_THYROID_QUESTIONS)

    async def list_questions(self) -> list[dict]:
        return await self._get_question_docs()

    async def get_question(self, q_id: str):
        for question in await self._get_question_docs():
            if question.get("id") == q_id:
                return question
        return None

    async def _get_user_record(self, user_id: str) -> Optional[dict]:
        if self.responses_collection is None:
            return self._memory_responses.get(user_id)

        try:
            return await self.responses_collection.find_one({"user_id": user_id}, _NO_ID)
        except Exception:
            return self._memory_responses.get(user_id)

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _parse_input_value(selected_option_ids: list[str]) -> float:
        if not selected_option_ids:
            return 0.0
        raw = selected_option_ids[0]
        if isinstance(raw, str) and raw.startswith("INPUT::"):
            raw = raw.split("::", 1)[1]
        return ThyroidService._to_float(raw, 0.0)

    @staticmethod
    def _is_truthy_response(selected_option_ids: list[str]) -> bool:
        if not selected_option_ids:
            return False
        raw = str(selected_option_ids[0]).strip().lower()
        return raw in {"1", "true", "yes", "y", "selected"} or raw.startswith("opt_")

    async def _validate_responses(self, responses: list, require_complete: bool = True) -> None:
        question_ids = [response.question_id for response in responses]
        if len(set(question_ids)) != len(question_ids):
            raise HTTPException(status_code=422, detail="Duplicate question_id entries found in submission")

        question_docs = await self._get_question_docs()
        required_question_ids = {doc["id"] for doc in question_docs}
        if not required_question_ids:
            raise HTTPException(status_code=400, detail="Thyroid questionnaire is not configured yet")

        if require_complete:
            missing_question_ids = required_question_ids - set(question_ids)
            if missing_question_ids:
                raise HTTPException(status_code=422, detail=f"Missing answers for question_id(s): {sorted(missing_question_ids)}")

        found_questions = {question["id"]: question for question in question_docs if question["id"] in question_ids}
        missing_qids = set(question_ids) - set(found_questions)
        if missing_qids:
            raise HTTPException(status_code=422, detail=f"Unknown question_id(s): {sorted(missing_qids)}")

        for response in responses:
            question = found_questions[response.question_id]
            q_type = question.get("q_type")
            selected_option_ids = response.selected_option_ids or []

            if q_type == "yes_no" and len(selected_option_ids) != 1:
                raise HTTPException(status_code=422, detail=f"Question '{response.question_id}' is yes_no; exactly one option must be selected")

            if q_type == "single_select" and len(selected_option_ids) > 1:
                raise HTTPException(status_code=422, detail=f"Question '{response.question_id}' is single_select; only one option may be selected")

            if q_type in ("yes_no", "single_select", "multi_select"):
                valid_option_ids = {option["id"] for option in (question.get("options") or []) if option.get("id")}
                if valid_option_ids:
                    invalid = [option_id for option_id in selected_option_ids if option_id not in valid_option_ids]
                    if invalid:
                        raise HTTPException(status_code=422, detail=f"Invalid option_id(s) {invalid} for question '{response.question_id}'")

            if q_type == "input":
                if len(selected_option_ids) != 1:
                    raise HTTPException(status_code=422, detail=f"Question '{response.question_id}' is input; exactly one numeric value must be submitted")
                value = selected_option_ids[0]
                if isinstance(value, str) and value.startswith("INPUT::"):
                    value = value.split("::", 1)[1]
                if self._to_float(value, None) is None:
                    raise HTTPException(status_code=422, detail=f"Question '{response.question_id}' requires a numeric value")

    async def save_user_responses(self, data: SubmitResponse):
        await self._validate_responses(data.responses, require_complete=True)
        payload = [response.model_dump() for response in data.responses]

        if self.responses_collection is None:
            existing = self._memory_responses.get(data.user_id, {})
            self._memory_responses[data.user_id] = {"user_id": data.user_id, "responses": payload, "updated_at": datetime.utcnow(), "created_at": existing.get("created_at", datetime.utcnow())}
            return {"status": "success", "message": "Thyroid responses saved"}

        try:
            await self.responses_collection.update_one(
                {"user_id": data.user_id},
                {"$set": {"responses": payload, "updated_at": datetime.utcnow()}, "$setOnInsert": {"created_at": datetime.utcnow()}},
                upsert=True,
            )
        except Exception:
            existing = self._memory_responses.get(data.user_id, {})
            self._memory_responses[data.user_id] = {"user_id": data.user_id, "responses": payload, "updated_at": datetime.utcnow(), "created_at": existing.get("created_at", datetime.utcnow())}

        return {"status": "success", "message": "Thyroid responses saved"}

    async def update_user_responses(self, data: PartialSubmitResponse):
        await self._validate_responses(data.responses, require_complete=False)
        existing_record = await self._get_user_record(data.user_id) or {"user_id": data.user_id, "responses": []}

        merged_by_question_id = {
            response.get("question_id"): response
            for response in existing_record.get("responses", [])
            if response.get("question_id")
        }

        for response in data.responses:
            merged_by_question_id[response.question_id] = response.model_dump()

        merged_responses = list(merged_by_question_id.values())

        if self.responses_collection is None:
            self._memory_responses[data.user_id] = {"user_id": data.user_id, "responses": merged_responses, "updated_at": datetime.utcnow(), "created_at": existing_record.get("created_at", datetime.utcnow())}
            return {"status": "success", "message": "Thyroid responses updated", "updated_count": len(data.responses), "total_responses": len(merged_responses)}

        try:
            await self.responses_collection.update_one(
                {"user_id": data.user_id},
                {"$set": {"responses": merged_responses, "updated_at": datetime.utcnow()}, "$setOnInsert": {"created_at": datetime.utcnow()}},
                upsert=True,
            )
        except Exception:
            self._memory_responses[data.user_id] = {"user_id": data.user_id, "responses": merged_responses, "updated_at": datetime.utcnow(), "created_at": existing_record.get("created_at", datetime.utcnow())}

        return {"status": "success", "message": "Thyroid responses updated", "updated_count": len(data.responses), "total_responses": len(merged_responses)}

    async def get_user_responses(self, user_id: str):
        record = await self._get_user_record(user_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"No thyroid responses found for user '{user_id}'")
        return record

    async def assert_user_completed_questionnaire(self, user_id: str) -> None:
        record = await self._get_user_record(user_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"No thyroid responses found for user '{user_id}'")

        question_docs = await self._get_question_docs()
        required_question_ids = {question["id"] for question in question_docs}
        answered_question_ids = {response["question_id"] for response in record.get("responses", [])}
        missing_question_ids = required_question_ids - answered_question_ids
        if missing_question_ids:
            raise HTTPException(status_code=422, detail=f"Prediction blocked: user has not completed all thyroid questionnaire questions. Missing: {sorted(missing_question_ids)}")

    async def get_user_answers_as_feature_dict(self, user_id: str) -> dict:
        record = await self._get_user_record(user_id)
        if not record:
            return {}

        feature_vector = {}
        all_question_docs = await self._get_question_docs()

        for question_doc in all_question_docs:
            for mapping in question_doc.get("direct_mappings") or []:
                feature_name = mapping.get("feature_name")
                if feature_name:
                    feature_vector[feature_name] = 0.0
            for option in question_doc.get("options") or []:
                for mapping in option.get("mappings") or []:
                    feature_name = mapping.get("feature_name")
                    if feature_name:
                        feature_vector[feature_name] = 0.0

        questions = {question["id"]: question for question in all_question_docs}

        for response in record.get("responses", []):
            question = questions.get(response.get("question_id"))
            if not question:
                continue

            q_type = question.get("q_type")
            selected_option_ids = response.get("selected_option_ids", [])

            if q_type == "input":
                input_value = self._parse_input_value(selected_option_ids)
                for mapping in question.get("direct_mappings") or []:
                    feature_name = mapping.get("feature_name")
                    if not feature_name:
                        continue
                    weight = self._to_float(mapping.get("feature_value", 1.0), 1.0)
                    feature_vector[feature_name] = max(self._to_float(feature_vector.get(feature_name, 0.0), 0.0), input_value * weight)
                continue

            options_by_id = {option.get("id"): option for option in (question.get("options") or []) if option.get("id")}

            if q_type == "yes_no" and not options_by_id and question.get("direct_mappings"):
                if self._is_truthy_response(selected_option_ids):
                    for mapping in question.get("direct_mappings") or []:
                        feature_name = mapping.get("feature_name")
                        if feature_name:
                            feature_vector[feature_name] = max(self._to_float(feature_vector.get(feature_name, 0.0), 0.0), self._to_float(mapping.get("feature_value", 1.0), 1.0))
                continue

            for option_id in selected_option_ids:
                option = options_by_id.get(option_id)
                if not option:
                    continue
                for mapping in option.get("mappings") or []:
                    feature_name = mapping.get("feature_name")
                    if not feature_name:
                        continue
                    value = self._to_float(mapping.get("feature_value", 0.0), 0.0)
                    current = self._to_float(feature_vector.get(feature_name, 0.0), 0.0)
                    feature_vector[feature_name] = max(current, value)

        return feature_vector

    async def validate_feature_coverage(self, user_id: str) -> dict:
        await self.assert_user_completed_questionnaire(user_id)
        features = await self.get_user_answers_as_feature_dict(user_id)
        question_docs = await self._get_question_docs()

        expected = set()
        for question_doc in question_docs:
            for mapping in question_doc.get("direct_mappings") or []:
                feature_name = mapping.get("feature_name")
                if feature_name:
                    expected.add(feature_name)
            for option in question_doc.get("options") or []:
                for mapping in option.get("mappings") or []:
                    feature_name = mapping.get("feature_name")
                    if feature_name:
                        expected.add(feature_name)

        return {"user_id": user_id, "mapped_feature_count": len(features), "expected_feature_count": len(expected), "features": sorted(features.keys())}

from copy import deepcopy
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.response import SubmitResponse
from fastapi import HTTPException
from datetime import datetime
from app.services.questionnaire_service import _OFFLINE_QUESTIONS

_NO_ID = {"_id": 0}


class ResponseService:
    _memory_responses: dict[str, dict] = {}

    def __init__(self, db: Optional[AsyncIOMotorDatabase]):
        self.collection = db.get_collection("user_responses") if db is not None else None
        self.questions_collection = db.get_collection("questions") if db is not None else None

    async def _get_question_docs(self) -> list[dict]:
        if self.questions_collection is None:
            return deepcopy(_OFFLINE_QUESTIONS)

        cursor = self.questions_collection.find({}, _NO_ID)
        return [doc async for doc in cursor]

    async def _get_user_record(self, user_id: str) -> Optional[dict]:
        if self.collection is None:
            return self._memory_responses.get(user_id)
        return await self.collection.find_one({"user_id": user_id}, _NO_ID)

    async def _validate_responses(self, responses: list) -> None:
        """Ensure all questionnaire questions are answered and options are valid."""
        question_ids = [r.question_id for r in responses]

        if len(set(question_ids)) != len(question_ids):
            raise HTTPException(
                status_code=422,
                detail="Duplicate question_id entries found in submission",
            )

        question_docs = await self._get_question_docs()
        required_question_ids = {doc["id"] for doc in question_docs}
        if not required_question_ids:
            raise HTTPException(
                status_code=400,
                detail="Questionnaire is not configured yet",
            )

        submitted_question_ids = set(question_ids)
        missing_question_ids = required_question_ids - submitted_question_ids
        if missing_question_ids:
            raise HTTPException(
                status_code=422,
                detail=f"Missing answers for question_id(s): {sorted(missing_question_ids)}",
            )

        found_questions = {q["id"]: q for q in question_docs if q["id"] in question_ids}

        missing_qids = set(question_ids) - set(found_questions)
        if missing_qids:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown question_id(s): {sorted(missing_qids)}",
            )

        for r in responses:
            q = found_questions[r.question_id]
            q_type = q.get("q_type")
            valid_opt_ids = {opt["id"] for opt in q.get("options", [])}

            if q_type == "yes_no" and len(r.selected_option_ids) != 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Question '{r.question_id}' is yes_no; "
                        "exactly one option must be selected"
                    ),
                )

            if q_type == "single_select" and len(r.selected_option_ids) > 1:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Question '{r.question_id}' is single_select; "
                        "only one option may be selected"
                    ),
                )

            if q_type in ("yes_no", "single_select", "multi_select"):
                bad_opts = [
                    oid for oid in r.selected_option_ids if oid not in valid_opt_ids
                ]
                if bad_opts:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid option_id(s) {bad_opts} for question '{r.question_id}'",
                    )

    async def assert_user_completed_questionnaire(self, user_id: str) -> None:
        record = await self._get_user_record(user_id)
        if not record:
            raise HTTPException(
                status_code=404, detail=f"No responses found for user '{user_id}'"
            )

        question_docs = await self._get_question_docs()
        required_question_ids = {doc["id"] for doc in question_docs}
        answered_question_ids = {r["question_id"] for r in record.get("responses", [])}

        missing_question_ids = required_question_ids - answered_question_ids
        if missing_question_ids:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Prediction blocked: user has not completed all questionnaire "
                    f"questions. Missing: {sorted(missing_question_ids)}"
                ),
            )

    async def save_user_responses(self, data: SubmitResponse):
        await self._validate_responses(data.responses)
        if self.collection is None:
            existing = self._memory_responses.get(data.user_id, {})
            self._memory_responses[data.user_id] = {
                "user_id": data.user_id,
                "responses": [r.model_dump() for r in data.responses],
                "updated_at": datetime.utcnow(),
                "created_at": existing.get("created_at", datetime.utcnow()),
            }
            return {"status": "success", "message": "Responses saved"}

        await self.collection.update_one(
            {"user_id": data.user_id},
            {
                "$set": {
                    "responses": [r.model_dump() for r in data.responses],
                    "updated_at": datetime.utcnow(),
                },
                "$setOnInsert": {"created_at": datetime.utcnow()},
            },
            upsert=True,
        )
        return {"status": "success", "message": "Responses saved"}

    async def get_user_responses(self, user_id: str):
        record = await self._get_user_record(user_id)
        if not record:
            raise HTTPException(
                status_code=404, detail=f"No responses found for user '{user_id}'"
            )
        return record

    async def get_user_answers_as_feature_dict(self, user_id: str) -> dict:
        record = await self._get_user_record(user_id)
        if not record:
            return {}

        # Initialize all known questionnaire-mapped features to 0.0 so that
        # non-selected options are treated as valid zero values, not missing values.
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

        # Build a question lookup so we can correctly process both option mappings
        # and direct input mappings for each answered question.
        response_question_ids = [r["question_id"] for r in record.get("responses", [])]
        questions = {
            q["id"]: q
            for q in all_question_docs
            if q["id"] in response_question_ids
        }

        def _to_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return default

        def _parse_input_value(selected_option_ids):
            if not selected_option_ids:
                return 0.0
            raw = selected_option_ids[0]
            # Stored format: INPUT::<numeric>
            if isinstance(raw, str) and raw.startswith("INPUT::"):
                return _to_float(raw.split("::", 1)[1], 0.0)
            return _to_float(raw, 0.0)

        for response in record.get("responses", []):
            question = questions.get(response.get("question_id"))
            if not question:
                continue

            q_type = question.get("q_type")
            selected_option_ids = response.get("selected_option_ids", [])

            if q_type == "input":
                input_value = _parse_input_value(selected_option_ids)
                for mapping in question.get("direct_mappings") or []:
                    feature_name = mapping.get("feature_name")
                    weight = _to_float(mapping.get("feature_value", 1.0), 1.0)
                    value = input_value * weight
                    current = _to_float(feature_vector.get(feature_name, 0.0), 0.0)
                    feature_vector[feature_name] = max(current, value)
                continue

            options_by_id = {
                opt.get("id"): opt for opt in (question.get("options") or []) if opt.get("id")
            }
            for option_id in selected_option_ids:
                option = options_by_id.get(option_id)
                if not option:
                    continue
                for mapping in option.get("mappings") or []:
                    feature_name = mapping.get("feature_name")
                    value = _to_float(mapping.get("feature_value", 0.0), 0.0)
                    current = _to_float(feature_vector.get(feature_name, 0.0), 0.0)
                    feature_vector[feature_name] = max(current, value)

        return feature_vector
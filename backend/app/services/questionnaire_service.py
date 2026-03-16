from copy import deepcopy
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.questionnaire import QuestionCreate, QuestionUpdate
from fastapi import HTTPException
import uuid

# Exclude _id from every MongoDB read — single source of truth
_NO_ID = {"_id": 0}
_OFFLINE_QUESTION_CATALOG = [
    (
        "menstrual",
        [
            ("How regular is your menstrual cycle?", ["Very Regular (28+/-2 days)", "Somewhat Regular", "Irregular", "Very Irregular"], ["irregular or missed periods", "Irregular / Missed periods", "Cycle Length"]),
            ("How would you rate your period pain?", ["No Pain", "Mild - manageable", "Moderate - affects daily life", "Severe - debilitating"], ["Menstrual pain (Dysmenorrhea)", "Painful cramps during period", "Pelvic pain"]),
            ("How heavy is your flow on heaviest days?", ["Light", "Moderate", "Heavy", "Very Heavy (clots)"], ["Heavy / Extreme menstrual bleeding", "Abnormal uterine bleeding"]),
            ("Do you experience spotting between periods?", ["Never", "Rarely", "Sometimes", "Often"], ["Abnormal uterine bleeding", "Constant bleeding"]),
            ("How long does your period usually last?", ["1-3 days", "4-5 days", "6-7 days", "More than 7 days"], ["Period Length"]),
        ],
    ),
    (
        "pcos",
        [
            ("Have you noticed unexplained weight gain recently?", ["No", "Slight gain (< 3 kg)", "Moderate gain (3-8 kg)", "Significant gain (> 8 kg)"], ["Overweight"]),
            ("Do you experience excessive hair growth (face, chest, back)?", ["Not at all", "Slightly", "Moderately", "Significantly"], ["Hair growth on Chin", "Hair growth  on Cheeks", "Hair growth Between breasts", "Hair growth  on Upper lips", "Hair growth in Arms", "Hair growth on Inner thighs"]),
            ("How is your acne/skin condition?", ["Clear skin", "Occasional breakouts", "Frequent breakouts", "Severe/persistent acne"], ["Acne or skin tags"]),
            ("Have you noticed hair thinning or loss on the scalp?", ["No", "Slight thinning", "Moderate thinning", "Significant hair loss"], ["Hair thinning or hair loss"]),
            ("Do you experience mood swings or depression?", ["Rarely", "Occasionally", "Frequently", "Almost always"], ["always tired", "Fatigue / Chronic fatigue"]),
        ],
    ),
    (
        "thyroid",
        [
            ("How are your energy levels throughout the day?", ["High energy", "Moderate energy", "Often tired", "Constantly exhausted"], ["always tired", "Fatigue / Chronic fatigue"]),
            ("How is your sensitivity to temperature?", ["Normal", "Feel cold easily", "Feel hot easily", "Extreme sensitivity"], ["Malaise / Sickness"]),
            ("Have you noticed changes in your weight without diet changes?", ["No change", "Slight gain", "Significant gain", "Unexplained weight loss"], ["Overweight"]),
            ("Do you experience dry skin or hair?", ["No", "Occasionally", "Frequently", "Severely"], ["Dark patches"]),
            ("How is your heart rate on a typical day?", ["Normal", "Occasionally fast", "Often rapid", "Slow/sluggish feeling"], ["Pain / Chronic pain"]),
        ],
    ),
    (
        "mental",
        [
            ("How would you rate your stress levels this week?", ["Very low", "Manageable", "Quite stressed", "Overwhelmed"], ["exercise per week"]),
            ("How has your sleep quality been?", ["Excellent (7-9 hrs)", "Good (6-7 hrs)", "Poor (< 6 hrs)", "Very poor / insomnia"], ["always tired"]),
            ("How often have you felt anxious or worried?", ["Rarely", "Occasionally", "Frequently", "Almost constantly"], ["Pain / Chronic pain"]),
            ("How would you describe your overall mood?", ["Happy and positive", "Neutral", "Often sad/low", "Depressed"], ["Loss of appetite"]),
            ("Do you feel supported by people around you?", ["Very supported", "Mostly supported", "Somewhat isolated", "Very lonely"], ["relocated city"]),
        ],
    ),
    (
        "nutrition",
        [
            ("How balanced is your daily diet?", ["Very balanced", "Mostly balanced", "Often unhealthy", "Very poor diet"], ["eat outside per week"]),
            ("How much water do you drink daily?", ["> 2.5 L", "1.5-2.5 L", "0.5-1.5 L", "< 0.5 L"], ["always tired"]),
            ("How often do you exercise per week?", ["5+ times", "3-4 times", "1-2 times", "Rarely / never"], ["exercise per week"]),
            ("How much processed / junk food do you consume?", ["Very rarely", "Occasionally", "Frequently", "Daily"], ["canned food often", "eat outside per week"]),
            ("Do you take vitamins or supplements?", ["Yes, regularly", "Sometimes", "Rarely", "Never"], ["Hormonal problems"]),
        ],
    ),
    (
        "bone",
        [
            ("Do you experience joint pain or stiffness?", ["Never", "Occasionally", "Frequently", "Daily"], ["Pain / Chronic pain", "Painful bowel movements"]),
            ("How much dairy / calcium-rich food do you consume?", ["Daily", "A few times/week", "Rarely", "Never"], ["eat outside per week"]),
            ("Do you get adequate sunlight for Vitamin D?", ["Yes, daily", "A few times/week", "Rarely", "Almost never"], ["always tired"]),
            ("Do you have a family history of osteoporosis?", ["No known history", "Possibly", "Yes", "Not sure"], ["Fertility Issues"]),
            ("Do you experience back pain or frequent fractures?", ["Never", "Occasionally", "Frequently", "Ongoing issues"], ["Pelvic pain", "Painful cramps during period"]),
        ],
    ),
]


def _build_offline_questions() -> list[dict]:
    questions: list[dict] = []
    priority = 1

    for category, category_questions in _OFFLINE_QUESTION_CATALOG:
        for q_idx, (question_text, options, mapped_features) in enumerate(category_questions, start=1):
            qid = f"Q_{category.upper()}_{q_idx:02d}"
            built_options = []
            max_idx = max(len(options) - 1, 1)

            for o_idx, option_text in enumerate(options, start=1):
                severity = (o_idx - 1) / max_idx
                built_options.append(
                    {
                        "id": f"OPT_{qid}_{o_idx:02d}",
                        "text": option_text,
                        "description": None,
                        "mappings": [
                            {
                                "feature_name": feature_name,
                                "feature_value": float(severity),
                            }
                            for feature_name in mapped_features
                        ],
                    }
                )

            questions.append(
                {
                    "id": qid,
                    "text": question_text,
                    "category": category,
                    "q_type": "single_select",
                    "is_initial": True,
                    "priority": priority,
                    "options": built_options,
                }
            )
            priority += 1

    return questions


_OFFLINE_QUESTIONS = _build_offline_questions()


class QuestionnaireService:
    def __init__(self, db: Optional[AsyncIOMotorDatabase]):
        self.collection = db.get_collection("questions") if db is not None else None

    async def create_smart_question(self, data: QuestionCreate):
        if self.collection is None:
            raise HTTPException(status_code=503, detail="Questionnaire editor requires database")

        existing = await self.collection.find_one({"id": data.id}, _NO_ID)
        if existing:
            raise HTTPException(
                status_code=400, detail=f"Question with id '{data.id}' already exists"
            )

        question_dict = data.model_dump()

        # Pop direct_mappings — used only to build the yes_no option, must not be persisted
        direct_mappings = question_dict.pop("direct_mappings", None)

        if question_dict.get("options") is None:
            question_dict["options"] = []

        # Auto-generate the Yes option for yes_no questions
        if data.q_type == "yes_no" and direct_mappings:
            question_dict["options"] = [
                {
                    "id": f"OPT_{data.id}_YES",
                    "text": "Yes",
                    "description": "Select if this applies to you",
                    "mappings": direct_mappings,
                }
            ]

        # Auto-assign IDs to options that were submitted without one
        for opt in question_dict["options"]:
            if not opt.get("id"):
                opt["id"] = f"OPT_{uuid.uuid4().hex[:8].upper()}"

        await self.collection.insert_one(question_dict)
        question_dict.pop("_id", None)  # Motor adds _id in-place after insert
        return question_dict

    async def list_all_questions(self):
        if self.collection is None:
            return deepcopy(_OFFLINE_QUESTIONS)

        cursor = self.collection.find({}, _NO_ID).sort("priority", 1)
        questions = await cursor.to_list(length=200)
        for q in questions:
            if q.get("options") is None:
                q["options"] = []
        return questions

    async def get_question(self, q_id: str):
        if self.collection is None:
            for question in _OFFLINE_QUESTIONS:
                if question["id"] == q_id:
                    return deepcopy(question)
            return None

        q = await self.collection.find_one({"id": q_id}, _NO_ID)
        if q and q.get("options") is None:
            q["options"] = []
        return q

    async def update_question(self, q_id: str, data: QuestionUpdate):
        if self.collection is None:
            raise HTTPException(status_code=503, detail="Questionnaire editor requires database")

        update_data = data.model_dump(exclude_unset=True)
        result = await self.collection.update_one({"id": q_id}, {"$set": update_data})
        if result.matched_count == 0:
            return None
        return await self.get_question(q_id)

    async def delete_question(self, q_id: str):
        if self.collection is None:
            raise HTTPException(status_code=503, detail="Questionnaire editor requires database")

        result = await self.collection.delete_one({"id": q_id})
        return result.deleted_count > 0
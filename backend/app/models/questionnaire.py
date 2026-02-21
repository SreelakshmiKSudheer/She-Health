from app.config.database import get_database
import uuid
from datetime import datetime


# Create Question
async def create_question(question_data: dict):
    db = get_database()
    question_data["question_id"] = str(uuid.uuid4())

    await db.questions.insert_one(question_data)
    return question_data


# Create Answer Option
async def create_option(option_data: dict):
    db = get_database()
    option_data["option_id"] = str(uuid.uuid4())

    await db.answer_options.insert_one(option_data)
    return option_data


# Save User Response
async def save_user_response(response_data: dict):
    db = get_database()
    response_data["response_id"] = str(uuid.uuid4())
    response_data["timestamp"] = datetime.utcnow()

    await db.user_responses.insert_one(response_data)
    return response_data

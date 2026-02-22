import sys
import os

# Add the parent directory to sys.path so we can import 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from app.db.database import SessionLocal, init_db
from app.services.questionnaire_service import QuestionnaireService
from app.schemas.questionnaire import QuestionCreate, OptionCreate, FeatureMapping

def seed_data():
    # 1. Initialize the database tables
    init_db()
    
    db: Session = SessionLocal()
    service = QuestionnaireService(db)

    # Define your 31 grouped questions here
    # This is an example of a Grouped Multi-Select Question
    questions_to_seed = [
        QuestionCreate(
            id="Q_PCOS_SYMPTOMS",
            text="Have you noticed any of the following skin or hair changes?",
            category="PCOS",
            q_type="multi_select",
            priority=1,
            options=[
                OptionCreate(
                    text="Excessive facial/body hair (Hirsutism)",
                    mappings=[FeatureMapping(feature_name="feature_hirsutism", feature_value=1.0)]
                ),
                OptionCreate(
                    text="Severe Acne or oily skin",
                    mappings=[FeatureMapping(feature_name="feature_acne", feature_value=1.0)]
                ),
                OptionCreate(
                    text="Thinning hair on the head",
                    mappings=[FeatureMapping(feature_name="feature_hair_thinning", feature_value=1.0)]
                )
            ]
        ),
        # Example of a Yes/No Question (Directly maps to 1 feature)
        QuestionCreate(
            id="Q_THYROID_FATIGUE",
            text="Do you experience unexplained chronic fatigue?",
            category="Thyroid",
            q_type="yes_no",
            priority=2,
            direct_mappings=[FeatureMapping(feature_name="feature_thyroid_fatigue", feature_value=1.0)]
        ),
        # Example of a Single Select with multiple feature weights
        QuestionCreate(
            id="Q_CYCLE_REGULARITY",
            text="How regular are your menstrual cycles?",
            category="General",
            q_type="single_select",
            priority=3,
            options=[
                OptionCreate(
                    text="Regular (21-35 days)",
                    mappings=[FeatureMapping(feature_name="feature_irregular_cycle", feature_value=0.0)]
                ),
                OptionCreate(
                    text="Irregular",
                    mappings=[FeatureMapping(feature_name="feature_irregular_cycle", feature_value=1.0)]
                )
            ]
        )
    ]

    print("🌱 Starting seeding process...")
    for q_data in questions_to_seed:
        # Check if question already exists to avoid duplicates
        if not service.get_question(q_data.id):
            service.create_smart_question(q_data)
            print(f"✅ Seeded: {q_data.id}")
        else:
            print(f"⏩ Skipped (Already exists): {q_data.id}")

    db.close()
    print("✨ Seeding complete!")

if __name__ == "__main__":
    seed_data()
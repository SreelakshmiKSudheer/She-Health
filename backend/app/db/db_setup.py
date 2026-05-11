# ═══════════════════════════════════════════════════════════════════════════
# db_setup.py  –  She Health · MongoDB Collection Setup
#
# Run ONCE to create indexes:
#   python db_setup.py
#
# Also contains the Pydantic/Motor document schema for reference.
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb://localhost:27017"   # or your Atlas SRV string
DB_NAME   = "shedhealth_db"


async def setup():
    client = AsyncIOMotorClient(MONGO_URI)
    db     = client[DB_NAME]

    # ── cycle_logs collection ────────────────────────────────────────────────
    # Document shape:
    # {
    #   "_id":               ObjectId,
    #   "user_id":           str,          ← matches the user's _id string
    #   "cycle_start_date":  datetime,     ← first day of bleeding this cycle
    #   "cycle_length":      int,          ← e.g. 28  (from Q_CYCLE_LENGTH)
    #   "period_duration":   int,          ← e.g. 5   (from Q_PERIOD_DURATION)
    #   "created_at":        datetime,
    # }

    cycle_col = db["cycle_logs"]
    await cycle_col.create_index(
        [("user_id", 1), ("cycle_start_date", -1)],
        name="user_cycle_date",
        unique=True,          # prevent duplicate entries for same start date
    )
    await cycle_col.create_index("user_id", name="user_id_idx")
    print("✅  cycle_logs indexes created")

    # ── questionnaire_responses collection ───────────────────────────────────
    # Document shape (simplified; your existing schema may differ):
    # {
    #   "_id":          ObjectId,
    #   "user_id":      str,
    #   "submitted_at": datetime,
    #   "answers": {
    #     "Q_CYCLE_LENGTH":   "28",
    #     "Q_PERIOD_DURATION": "5",
    #     "Q_CYCLE_DESC":     "Regular",
    #     ...all other question IDs as keys...
    #   }
    # }

    qr_col = db["questionnaire_responses"]
    await qr_col.create_index(
        [("user_id", 1), ("submitted_at", -1)],
        name="user_submitted",
    )
    print("✅  questionnaire_responses indexes created")

    client.close()
    print("🎉  Database setup complete.")


if __name__ == "__main__":
    asyncio.run(setup())
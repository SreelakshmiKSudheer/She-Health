from fastapi import APIRouter, Body
try:
    from app.db.database import MongoDB
except ModuleNotFoundError:
    from ..db.database import MongoDB
from datetime import datetime

router = APIRouter(prefix="/cycle", tags=["Cycle Tracking"])

@router.post("/log")
async def log_daily_data(data: dict = Body(...)):
    """
    Expects: { "user_id": "...", "date": "2024-03-20", "symptoms": ["cramps"], "flow": "medium" }
    """
    col = MongoDB.get_collection("cycle_logs")

    flow = str(data.get("flow", "")).strip().lower()
    provided_is_period = data.get("is_period")
    if provided_is_period is None:
        # If flow indicates any bleeding, flag as a period day.
        data["is_period"] = flow in {"light", "medium", "heavy", "normal", "spotting"}
    else:
        data["is_period"] = bool(provided_is_period)

    # Normalize date to YYYY-MM-DD string
    if "date" in data:
        try:
            date_obj = datetime.fromisoformat(str(data["date"]))
            data["date"] = date_obj.date().isoformat()
        except Exception:
            pass

    # Upsert: Update if date exists, else insert
    await col.update_one(
        {"user_id": data["user_id"], "date": data["date"]},
        {"$set": data},
        upsert=True
    )
    return {"status": "recorded"}

@router.get("/predictions/{user_id}")
async def get_predictions(user_id: str):
    col = MongoDB.get_collection("cycle_logs")
    
    # Get all period logs ordered by date desc (most recent first)
    period_entries = await col.find(
        {"user_id": user_id, "is_period": True}
    ).sort("date", -1).to_list(length=365)

    if not period_entries:
        # fallback: if no explicit period logs, take the latest entered date as starting point.
        latest_entry = await col.find_one(
            {"user_id": user_id},
            sort=[("date", -1)]
        )
        if not latest_entry:
            return {
                "status": "error",
                "message": "No cycle logs found. Please log your period dates first."
            }
        last_period_date = latest_entry["date"]
        cycle_length = 28
    else:
        last_period_date = period_entries[0]["date"]

        # If we have at least two period entries, estimate cycle length from history.
        if len(period_entries) >= 2:
            dates = []
            try:
                dates = [datetime.fromisoformat(entry["date"]).date() for entry in period_entries]
            except Exception:
                dates = []
            if len(dates) >= 2:
                deltas = [
                    (dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)
                ]
                cycle_length = max(21, min(35, round(sum(deltas) / len(deltas))))
            else:
                cycle_length = 28
        else:
            cycle_length = 28

    from app.ml.predictor import CyclePredictor
    predictions = CyclePredictor.calculate_predictions(last_period_date, cycle_length=cycle_length)

    days_until = (datetime.fromisoformat(predictions["next_period"]) - datetime.now()).days

    predictions.update({
        "user_id": user_id,
        "days_until_next_period": max(0, days_until),
        "last_logged_date": last_period_date,
        "cycle_length_used": cycle_length,
        "period_entries_count": len(period_entries),
    })

    return predictions
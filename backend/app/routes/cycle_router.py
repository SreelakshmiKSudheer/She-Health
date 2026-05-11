# ═══════════════════════════════════════════════════════════════════════════
# cycle_router.py  –  She Health · Cycle Logging & History
#
# Endpoints used by Flutter frontend:
#   POST   /api/cycle/log          → save / upsert a cycle
#   GET    /api/cycle/history      → fetch history (History tab + ring calendar)
#   DELETE /api/cycle/{id}         → remove a cycle entry
#
# Notes:
#   • period_duration is always stored as DEFAULT_PERIOD_DURATION (5).
#     It is never taken from the user — fixed server-side.
#   • unusual_bleeding is accepted in the payload but silently discarded;
#     it is used only by the ML prediction router, never stored here.
# ═══════════════════════════════════════════════════════════════════════════

from datetime import datetime, timedelta
from typing import List, Optional

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, validator
import os

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME", "she_health_db")

DEFAULT_PERIOD_DURATION = 5

_client: Optional[AsyncIOMotorClient] = None

def _db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client[DB_NAME]

def _col():
    return _db()["cycle_logs"]


# ── Models ────────────────────────────────────────────────────────────────────

class CycleLogIn(BaseModel):
    user_id:          str
    cycle_start_date: datetime
    cycle_length:     int  = Field(ge=15, le=90)
    is_historical:    bool = False
    # Always overridden to DEFAULT_PERIOD_DURATION server-side
    period_duration:  int  = Field(default=DEFAULT_PERIOD_DURATION, ge=1, le=20)
    # Accepted to avoid 422 but discarded — ML router uses this, not us
    unusual_bleeding: Optional[bool] = None

    @validator("cycle_start_date", pre=True)
    def _parse_date(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @validator("period_duration", always=True)
    def _fix_duration(cls, v):
        # Store whatever the user entered, clamped to valid range
        return max(1, min(20, v))


class CycleLogOut(BaseModel):
    id:                   str
    user_id:              str
    cycle_start_date:     str
    cycle_length:         int
    period_duration:      int
    is_historical:        bool
    next_period_date:     str
    ovulation_date:       str
    fertile_window_start: str
    fertile_window_end:   str
    pms_window_start:     str
    created_at:           str

    class Config:
        from_attributes = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _derive_dates(start: datetime, cycle_length: int) -> dict:
    """Compute all phase calendar dates from cycle start + length."""
    return {
        "next_period_date":     (start + timedelta(days=cycle_length)).date().isoformat(),
        "ovulation_date":       (start + timedelta(days=cycle_length - 14)).date().isoformat(),
        "fertile_window_start": (start + timedelta(days=cycle_length - 19)).date().isoformat(),
        "fertile_window_end":   (start + timedelta(days=cycle_length - 14)).date().isoformat(),
        "pms_window_start":     (start + timedelta(days=cycle_length - 5)).date().isoformat(),
    }


def _to_out(doc: dict) -> CycleLogOut:
    dates = _derive_dates(doc["cycle_start_date"], doc["cycle_length"])
    return CycleLogOut(
        id               = str(doc["_id"]),
        user_id          = doc["user_id"],
        cycle_start_date = doc["cycle_start_date"].date().isoformat(),
        cycle_length     = doc["cycle_length"],
        period_duration  = doc.get("period_duration", DEFAULT_PERIOD_DURATION),
        is_historical    = doc.get("is_historical", False),
        created_at       = doc.get("created_at", datetime.utcnow()).isoformat(),
        **dates,
    )


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter()


# ── POST /api/cycle/log ───────────────────────────────────────────────────────
@router.post("/log", response_model=CycleLogOut, status_code=201)
async def log_cycle(p: CycleLogIn):
    """
    Save or upsert a cycle entry.
    Stored fields: user_id, cycle_start_date, cycle_length, is_historical.
    period_duration is always fixed at 5 regardless of input.
    unusual_bleeding is NOT stored here — the ML router receives it directly.
    Returns all derived phase dates so Flutter can render the ring immediately.
    """
    doc = {
        "user_id":          p.user_id,
        "cycle_start_date": p.cycle_start_date,
        "cycle_length":     p.cycle_length,
        "period_duration":  DEFAULT_PERIOD_DURATION,
        "is_historical":    p.is_historical,
        "created_at":       datetime.utcnow(),
    }
    result = await _col().find_one_and_update(
        {"user_id": p.user_id, "cycle_start_date": p.cycle_start_date},
        {"$set": doc},
        upsert=True,
        return_document=True,
    )
    if result is None:
        result = await _col().find_one(
            {"user_id": p.user_id, "cycle_start_date": p.cycle_start_date}
        )
    return _to_out(result)


# ── GET /api/cycle/history ────────────────────────────────────────────────────
@router.get("/history", response_model=List[CycleLogOut])
async def get_cycle_history(
    user_id: str = Query(...),
    limit:   int = Query(default=24, ge=1, le=365),
):
    """
    Return cycle history sorted newest-first (default: last 24 cycles).
    Used by:
      • App startup  — populates ring calendar with period/fertile/PMS days
      • History tab  — renders the list of past cycles
      • Insights tab — cycle length trend chart + statistics
    Each entry includes all derived phase dates (next_period_date,
    ovulation_date, fertile_window_start/end, pms_window_start).
    """
    docs = await _col().find(
        {"user_id": user_id},
        sort=[("cycle_start_date", -1)],
    ).limit(limit).to_list(limit)

    return [_to_out(d) for d in docs]


# ── DELETE /api/cycle/{id} ────────────────────────────────────────────────────
@router.delete("/{id}", status_code=204)
async def delete_cycle(id: str, user_id: str = Query(...)):
    """
    Delete a single cycle by its MongoDB _id.
    user_id is required as a query param for ownership verification.
    Returns 404 if the cycle is not found or belongs to a different user.
    """
    r = await _col().delete_one(
        {"_id": ObjectId(id), "user_id": user_id}
    )
    if r.deleted_count == 0:
        raise HTTPException(404, "Cycle not found or does not belong to this user")


# ── Models for daily symptom log ──────────────────────────────────────────────

class DailyLogIn(BaseModel):
    user_id:  str
    date:     str
    flow:     int        = 0   # 0=none 1=light 2=medium 3=heavy
    moods:    List[str]  = []
    symptoms: List[str]  = []
    note:     str        = ""

class DailyLogOut(BaseModel):
    id:       str
    user_id:  str
    date:     str
    flow:     int
    moods:    List[str]
    symptoms: List[str]
    note:     str

    class Config:
        from_attributes = True


def _daily():
    return _db()["cycle_daily_logs"]

def _daily_out(doc: dict) -> DailyLogOut:
    return DailyLogOut(
        id       = str(doc["_id"]),
        user_id  = doc["user_id"],
        date     = doc["date"],
        flow     = doc.get("flow", 0),
        moods    = doc.get("moods", []),
        symptoms = doc.get("symptoms", []),
        note     = doc.get("note", ""),
    )


# ── POST /api/cycle/daily-log ─────────────────────────────────────────────────
@router.post("/daily-log", response_model=DailyLogOut, status_code=201)
async def save_daily_log(p: DailyLogIn):
    """Save or update a daily symptom log entry."""
    doc = {
        "user_id":    p.user_id,
        "date":       p.date,
        "flow":       p.flow,
        "moods":      p.moods,
        "symptoms":   p.symptoms,
        "note":       p.note,
        "updated_at": datetime.utcnow(),
    }
    result = await _db()["cycle_daily_logs"].find_one_and_update(
        {"user_id": p.user_id, "date": p.date},
        {"$set": doc},
        upsert=True,
        return_document=True,
    )
    if result is None:
        result = await _db()["cycle_daily_logs"].find_one(
            {"user_id": p.user_id, "date": p.date}
        )
    return _daily_out(result)
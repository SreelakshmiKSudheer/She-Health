# ═══════════════════════════════════════════════════════════════════════════
# cycle_prediction_router.py  –  She Health · ML Cycle Prediction
#
# Endpoints:
#   POST  /api/cycle-predict/predict  → predict for a single cycle
#                                       (called after logging current cycle)
#   POST  /api/cycle-predict/bulk     → predict for multiple past cycles
#                                       (called after adding past cycles
#                                        in the History tab)
#
# ML model: DecisionTreeRegressor trained on FedCycle dataset.
# Predicts EstimatedDayOfOvulation → derives all phase dates from it.
# Falls back to Hartman formula (cycle_length - 14) if model file missing.
#
# User supplies only 3 values per cycle:
#   cycle_start_date  — Period Start Date field
#   length_of_cycle   — Cycle Length stepper
#   unusual_bleeding  — Unusual Bleeding toggle (0 or 1)
# All other ML features use fixed defaults (age=25, bmi=22, etc.)
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME", "she_health_db")

_client: Optional[AsyncIOMotorClient] = None

def _db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client[DB_NAME]


# ── ML model ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv(
    "CYCLE_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "ml", "cycle_model.pkl"),
)

_model = None

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_cycle_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# ── Fixed defaults ────────────────────────────────────────────────────────────
DEFAULT_AGE             = 25
DEFAULT_BMI             = 22.0
DEFAULT_PERIOD_DURATION = 5


# ── Pydantic models ───────────────────────────────────────────────────────────

class CyclePredictionIn(BaseModel):
    """
    Per-cycle prediction input.
    Three fields come from the user; everything else uses fixed defaults.
    """
    user_id: str = ""  # optional in nested bulk cycles, set by parent

    # ── User-supplied ─────────────────────────────────────────────────────
    cycle_start_date: str = Field(description="ISO date e.g. 2026-05-01")
    length_of_cycle:  int = Field(default=28, ge=15, le=90)
    unusual_bleeding: int = Field(default=0,  ge=0,  le=1,
                                   description="0=No  1=Yes")

    # ── Fixed defaults / derived ─────────────────────────────────────────
    cycle_number:        int   = Field(default=1,                       ge=1)
    length_of_luteal:    int   = Field(default=14,                      ge=0, le=60)
    high_days:           int   = Field(default=5,                       ge=0, le=15)
    peak_days:           int   = Field(default=2,                       ge=0, le=5)
    phases_bleeding:     int   = Field(default=DEFAULT_PERIOD_DURATION, ge=0, le=20)
    intercourse_fertile: int   = Field(default=0,                       ge=0, le=1)
    age:                 int   = Field(default=DEFAULT_AGE,             ge=15, le=55)
    bmi:                 float = Field(default=DEFAULT_BMI,             ge=15.0, le=45.0)
    method:              int   = Field(default=1,                       ge=0, le=5)


class CyclePredictionOut(BaseModel):
    """
    Full prediction result for one cycle.
    Flutter uses the latest entry to populate the ML card (Log Day tab)
    and ML banner (Insights tab).
    """
    user_id:                 str
    cycle_start_date:        str
    length_of_cycle:         int
    predicted_ovulation_day: int    # Day number within cycle e.g. 14
    ovulation_date:          str    # Actual calendar date
    fertile_window_start:    str
    fertile_window_end:      str
    next_period_date:        str
    pms_window_start:        str
    luteal_phase_length:     int
    prediction_source:       str    # "ml_model" | "formula_fallback"
    confidence:              str    # "high" | "medium" | "low"


class BulkPredictionIn(BaseModel):
    """
    Bulk prediction input for multiple past cycles.
    Flutter sends this after the user adds past cycles in the History tab.
    Each cycle only needs: cycle_start_date, length_of_cycle, unusual_bleeding.
    cycle_number is auto-assigned based on list position.
    """
    user_id: str
    cycles:  list[CyclePredictionIn]


# ── Internal helpers ──────────────────────────────────────────────────────────

# Feature column names — must match exactly what the model was trained on
_FEATURE_COLS = [
    "CycleNumber", "LengthofCycle", "LengthofLutealPhase",
    "TotalNumberofHighDays", "TotalNumberofPeakDays", "UnusualBleeding",
    "PhasesBleeding", "IntercourseInFertileWindow", "Age", "BMI", "Method",
]

def _build_features(p: CyclePredictionIn):
    """Return a pandas DataFrame so the model gets named features (no warning)."""
    import pandas as pd
    return pd.DataFrame([[
        p.cycle_number,
        p.length_of_cycle,
        p.length_of_luteal,
        p.high_days,
        p.peak_days,
        p.unusual_bleeding,
        p.phases_bleeding,
        p.intercourse_fertile,
        p.age,
        int(p.bmi),
        p.method,
    ]], columns=_FEATURE_COLS, dtype=float)


def _formula_ovulation(cycle_length: int) -> int:
    return max(10, cycle_length - 14)


def _confidence(ovulation_day: int, cycle_length: int) -> str:
    diff = abs(ovulation_day - _formula_ovulation(cycle_length))
    if diff <= 2: return "high"
    if diff <= 4: return "medium"
    return "low"


def _build_response(
    p: CyclePredictionIn,
    ovulation_day: int,
    source: str,
) -> CyclePredictionOut:
    start = datetime.fromisoformat(p.cycle_start_date.replace("Z", ""))
    ovulation_day  = max(10, min(ovulation_day, p.length_of_cycle - 7))
    ovulation_date = start + timedelta(days=ovulation_day - 1)
    fertile_start  = ovulation_date - timedelta(days=5)
    next_period    = start + timedelta(days=p.length_of_cycle)
    pms_start      = next_period - timedelta(days=5)

    return CyclePredictionOut(
        user_id                 = p.user_id,
        cycle_start_date        = p.cycle_start_date,
        length_of_cycle         = p.length_of_cycle,
        predicted_ovulation_day = ovulation_day,
        ovulation_date          = ovulation_date.date().isoformat(),
        fertile_window_start    = fertile_start.date().isoformat(),
        fertile_window_end      = ovulation_date.date().isoformat(),
        next_period_date        = next_period.date().isoformat(),
        pms_window_start        = pms_start.date().isoformat(),
        luteal_phase_length     = p.length_of_cycle - ovulation_day,
        prediction_source       = source,
        confidence              = _confidence(ovulation_day, p.length_of_cycle),
    )


async def _run_prediction(p: CyclePredictionIn) -> tuple[CyclePredictionOut, int, str]:
    """Core prediction logic shared by /predict and /bulk."""
    p = p.model_copy(update={"length_of_luteal": p.length_of_cycle - 14})
    try:
        model         = _load_model()
        raw_pred      = model.predict(_build_features(p))[0]
        ovulation_day = max(10, int(round(raw_pred)))
        source        = "ml_model"
    except FileNotFoundError:
        ovulation_day = _formula_ovulation(p.length_of_cycle)
        source        = "formula_fallback"
    return _build_response(p, ovulation_day, source), ovulation_day, source


async def _persist(p: CyclePredictionIn, ovulation_day: int, source: str):
    """Save prediction to MongoDB for audit. Never raises."""
    try:
        await _db()["cycle_ml_predictions"].insert_one({
            "user_id":          p.user_id,
            "cycle_start_date": p.cycle_start_date,
            "input_features":   p.model_dump(),
            "ovulation_day":    ovulation_day,
            "source":           source,
            "created_at":       datetime.utcnow(),
        })
    except Exception:
        pass


# ── Router ────────────────────────────────────────────────────────────────────

# ── Weighted cycle length predictor ──────────────────────────────────────────

def _predict_next_cycle_length(cycle_lengths: list[int]) -> dict:
    """
    Predict the next cycle length from history using exponential weighting.
    Recent cycles count more than older ones.
    Returns: predicted_length, lower_bound, upper_bound, is_irregular, confidence
    """
    n = len(cycle_lengths)

    if n == 0:
        return {"predicted": 28, "lower": 25, "upper": 31,
                "is_irregular": False, "confidence": "low"}

    if n == 1:
        return {"predicted": cycle_lengths[0], "lower": cycle_lengths[0] - 3,
                "upper": cycle_lengths[0] + 3, "is_irregular": False, "confidence": "low"}

    # Use up to last 6 cycles, weight: most recent = highest weight
    recent = cycle_lengths[-6:]  # oldest to newest
    m = len(recent)
    # Weights: 1,2,3... m (newest = m)
    weights = list(range(1, m + 1))
    total_w = sum(weights)
    weighted_mean = sum(w * l for w, l in zip(weights, recent)) / total_w

    # Standard deviation (unweighted for simplicity)
    mean_unw = sum(recent) / m
    std = (sum((x - mean_unw) ** 2 for x in recent) / m) ** 0.5

    # Irregularity: std > 3 days is clinically meaningful
    is_irregular = std > 3.0

    # Prediction interval widens with irregularity
    margin = max(3, round(std * 1.5))

    predicted = round(weighted_mean)
    predicted = max(15, min(90, predicted))

    confidence = "high" if std <= 2 else "medium" if std <= 4 else "low"

    return {
        "predicted":    predicted,
        "lower":        max(15, predicted - margin),
        "upper":        min(90, predicted + margin),
        "std_dev":      round(std, 1),
        "is_irregular": is_irregular,
        "confidence":   confidence,
    }


def _build_response_with_history(
    p: CyclePredictionIn,
    ovulation_day: int,
    source: str,
    next_cycle_pred: dict,
) -> CyclePredictionOut:
    """
    Build the response using the PREDICTED next cycle length from history
    rather than the user-entered current cycle length.
    This makes the next_period_date adaptive to actual past patterns.
    """
    start = datetime.fromisoformat(p.cycle_start_date.replace("Z", ""))
    ovulation_day = max(10, min(ovulation_day, p.length_of_cycle - 7))

    ovulation_date = start + timedelta(days=ovulation_day - 1)
    fertile_start  = ovulation_date - timedelta(days=5)

    # Use predicted next cycle length for the next period date
    next_len       = next_cycle_pred["predicted"]
    next_period    = start + timedelta(days=next_len)
    pms_start      = next_period - timedelta(days=5)
    luteal         = p.length_of_cycle - ovulation_day

    # Confidence factors in both ML confidence and cycle regularity
    ml_conf = _confidence(ovulation_day, p.length_of_cycle)
    reg_conf = next_cycle_pred["confidence"]
    # Overall: take the worse of the two
    conf_rank = {"high": 2, "medium": 1, "low": 0}
    final_conf = min(conf_rank[ml_conf], conf_rank[reg_conf])
    final_conf_str = ["low", "medium", "high"][final_conf]

    return CyclePredictionOut(
        user_id                 = p.user_id,
        cycle_start_date        = p.cycle_start_date,
        length_of_cycle         = next_len,        # predicted, not user-entered
        predicted_ovulation_day = ovulation_day,
        ovulation_date          = ovulation_date.date().isoformat(),
        fertile_window_start    = fertile_start.date().isoformat(),
        fertile_window_end      = ovulation_date.date().isoformat(),
        next_period_date        = next_period.date().isoformat(),
        pms_window_start        = pms_start.date().isoformat(),
        luteal_phase_length     = luteal,
        prediction_source       = source,
        confidence              = final_conf_str,
    )


router = APIRouter()


# ── POST /api/cycle-predict/predict ───────────────────────────────────────────
@router.post("/predict", response_model=CyclePredictionOut)
async def predict_cycle(p: CyclePredictionIn):
    """
    Predict ovulation day (ML model) + next period date (weighted history).

    When the user has 2+ cycles in history, the next_period_date is computed
    from a weighted average of past cycle lengths (recent cycles weighted more),
    NOT from the user-entered cycle_length. This makes the prediction adapt to
    the user's actual irregular pattern instead of repeating the same interval.

    With only 1 cycle: falls back to user-entered cycle_length.
    """
    if not p.user_id:
        raise HTTPException(400, "user_id is required")

    # Step 1: Get ML ovulation prediction
    p_adj = p.model_copy(update={"length_of_luteal": p.length_of_cycle - 14})
    try:
        model         = _load_model()
        raw_pred      = model.predict(_build_features(p_adj))[0]
        ovulation_day = max(10, int(round(raw_pred)))
        source        = "ml_model"
        print(f"🔄 Running prediction for cycle starting at: {p.cycle_start_date}")
        print("✅ ML prediction successful.")
    except FileNotFoundError:
        ovulation_day = _formula_ovulation(p.length_of_cycle)
        source        = "formula_fallback"

    # Step 2: Fetch past cycle lengths from DB to predict NEXT cycle length
    past_docs = await _db()["cycle_logs"].find(
        {"user_id": p.user_id},
        sort=[("cycle_start_date", -1)]
    ).limit(6).to_list(6)

    past_lengths = [int(d["cycle_length"]) for d in past_docs if "cycle_length" in d]

    if len(past_lengths) >= 2:
        # Have history — use weighted predictor for next cycle length
        next_pred = _predict_next_cycle_length(list(reversed(past_lengths)))
        result    = _build_response_with_history(p, ovulation_day, source, next_pred)
    else:
        # First cycle — use standard response
        result, ovulation_day, source = await _run_prediction(p)

    await _persist(p, ovulation_day, source)
    return result


# ── POST /api/cycle-predict/bulk ──────────────────────────────────────────────
@router.post("/bulk", response_model=list[CyclePredictionOut])
async def bulk_predict(payload: BulkPredictionIn):
    """
    Predict for all cycles in the history list.

    For cycles 1..N-1: standard ML prediction using their own cycle_length.
    For cycle N (most recent): uses weighted average of ALL previous cycle
    lengths to predict the NEXT cycle length. This is the key difference
    from formula — if the user had cycles of 28, 35, 26 days, the predicted
    next period is not start+28 but start+weighted_avg(28,35,26) ≈ start+29,
    with the prediction window widening if cycles are irregular.

    The Flutter client uses the last result for the ML card.
    """
    if not payload.user_id:
        raise HTTPException(400, "user_id is required")
    if not payload.cycles:
        raise HTTPException(400, "cycles list must not be empty")

    all_lengths = [c.length_of_cycle for c in payload.cycles]
    results     = []

    for idx, cycle in enumerate(payload.cycles):
        # Always stamp parent user_id — cycle.user_id may be empty in nested payload
        cycle_meta = cycle.model_copy(update={
            "user_id":      payload.user_id,
            "cycle_number": idx + 1,
            # Derive luteal from cycle's own length
            "length_of_luteal": max(1, cycle.length_of_cycle - 14),
        })
        cycle_adj = cycle_meta.model_copy(
            update={"length_of_luteal": cycle_meta.length_of_cycle - 14})

        # ML ovulation prediction for every cycle
        try:
            model         = _load_model()
            raw_pred      = model.predict(_build_features(cycle_adj))[0]
            ovulation_day = max(10, int(round(raw_pred)))
            source        = "ml_model"
            print(f"🔄 Running prediction for cycle starting at: {cycle_meta.cycle_start_date}")
            print("✅ ML prediction successful.")
        except FileNotFoundError:
            ovulation_day = _formula_ovulation(cycle_meta.length_of_cycle)
            source        = "formula_fallback"

        is_last = (idx == len(payload.cycles) - 1)

        if is_last and len(all_lengths) >= 2:
            # Most recent cycle: predict NEXT cycle length from all previous lengths
            # Exclude the current cycle itself from the history used for prediction
            history_for_pred = all_lengths[:idx]  # all cycles before this one
            next_pred = _predict_next_cycle_length(history_for_pred)
            result    = _build_response_with_history(
                cycle_meta, ovulation_day, source, next_pred)
        else:
            # Earlier cycles: standard prediction using their own cycle_length
            result, ovulation_day, source = await _run_prediction(cycle_meta)

        await _persist(cycle_meta, ovulation_day, source)
        results.append(result)

    return results
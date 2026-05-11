# ═══════════════════════════════════════════════════════════════════════════
# train_cycle_model.py  –  She Health · Cycle Prediction Model Training
#
# Dataset: FedCycleData071012.csv
# Target:  EstimatedDayofOvulation
# Model:   DecisionTreeRegressor (R² ≈ 0.58, best from notebook comparison)
#
# Run:  python train_cycle_model.py --data path/to/FedCycleData071012.csv
#
# Output: ml/cycle_model.pkl  (loaded by cycle_prediction_router.py)
# ═══════════════════════════════════════════════════════════════════════════

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.preprocessing import LabelEncoder

# ── Features we actually use (available from questionnaire + cycle log) ───────
#
# SELECTED (available in She Health):
#   CycleNumber        → count of logged cycles in DB
#   LengthofCycle      → Q_CYCLE_LENGTH
#   LengthofLutealPhase→ derived as LengthofCycle - 14
#   TotalNumberofHighDays → default 5 (standard fertile window)
#   TotalNumberofPeakDays → default 2
#   UnusualBleeding    → Q_BLEEDING_PATTERNS (heavy/constant → 1)
#   PhasesBleeding     → Q_PERIOD_DURATION (proxy)
#   IntercourseInFertileWindow → default 0 (privacy)
#   Age                → Q_AGE / user profile
#   BMI                → user profile
#   Method             → derived from Q_CYCLE_DESC regularity
#
# EXCLUDED (not available / too sensitive):
#   ClientID, Group, MeanCycleLength, FirstDayofHigh, TotalHighPostPeak,
#   TotalDaysofFertility, TotalFertilityFormula, LengthofMenses,
#   MeanMensesLength, MensesScore*, MeanBleedingIntensity,
#   NumberofDaysofIntercourse, IntercourseDuringUnusBleed, AgeM,
#   Maristatus*, Yearsmarried, Wedding, Religion*, Ethnicity*,
#   Schoolyears*, Occupation*, Income*, Height, Weight, Reprocate,
#   Numberpreg, Livingkids, Miscarriages, Abortions, Medvits*,
#   Gynosurgeries, Boys, Girls, Urosurgeries, Breastfeeding,
#   Prevmethod, Methoddate, Whychart, Nextpreg*, Spousesame*,
#   Timeattemptpreg, ReproductiveCategory, CycleWithPeakorNot

FEATURE_COLS = [
    "CycleNumber",
    "LengthofCycle",
    "LengthofLutealPhase",
    "TotalNumberofHighDays",
    "TotalNumberofPeakDays",
    "UnusualBleeding",
    "PhasesBleeding",
    "IntercourseInFertileWindow",
    "Age",
    "BMI",
    "Method",
]
TARGET_COL = "EstimatedDayofOvulation"


def load_and_preprocess(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")

    # Select only needed columns
    cols = FEATURE_COLS + [TARGET_COL]
    df = df[cols].copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Convert all to numeric, coerce errors to NaN
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN with column mode
    for col in cols:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col].fillna(mode_val[0], inplace=True)
        else:
            df[col].fillna(0, inplace=True)

    # Convert to int (matches notebook)
    for col in cols:
        df[col] = df[col].fillna(0).astype(int)

    # Remove rows where target is 0 (invalid)
    df = df[df[TARGET_COL] > 0]

    # Clamp target to realistic ovulation range (day 10–25)
    df = df[(df[TARGET_COL] >= 8) & (df[TARGET_COL] <= 28)]

    print(f"Clean shape: {df.shape}")
    print(f"Target range: {df[TARGET_COL].min()} – {df[TARGET_COL].max()}")
    print(f"Target mean: {df[TARGET_COL].mean():.2f}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> DecisionTreeRegressor:
    print("\n── Splitting dataset (85% train / 15% test) ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Label encode any remaining object columns
    le = LabelEncoder()
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))

    y_train = y_train.astype(float)
    y_test  = y_test.astype(float)

    print("\n── Training DecisionTreeRegressor ──")
    model = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    mape   = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\n── Test Set Metrics ──")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y.astype(float),
                                cv=5, scoring="r2")
    print(f"\n── 5-Fold Cross-Validation R² ──")
    print(f"  Scores: {cv_scores.round(3)}")
    print(f"  Mean:   {cv_scores.mean():.4f}")
    print(f"  Std:    {cv_scores.std():.4f}")

    # Feature importance
    importances = pd.Series(model.feature_importances_,
                            index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n── Feature Importances ──")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    return model


def save_model(model, output_dir: str = "ml"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cycle_model.pkl")
    joblib.dump(model, path)
    print(f"\n✅ Model saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Train She Health cycle prediction model")
    parser.add_argument("--data", required=True,
                        help="Path to FedCycleData CSV file")
    parser.add_argument("--output", default="ml",
                        help="Output directory for model (default: ml/)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        return

    X, y = load_and_preprocess(args.data)
    model = train_and_evaluate(X, y)
    path  = save_model(model, args.output)

    # Quick sanity check
    print("\n── Sanity Check (sample predictions) ──")
    samples = pd.DataFrame([
        # cycle 28 days, age 25
        [1, 28, 14, 5, 2, 0, 5, 0, 25, 22, 1],
        # cycle 30 days, age 30
        [3, 30, 16, 6, 2, 0, 5, 0, 30, 24, 1],
        # cycle 26 days, unusual bleeding
        [2, 26, 12, 4, 1, 1, 5, 0, 28, 26, 0],
        # cycle 35 days (long)
        [1, 35, 21, 7, 2, 0, 4, 0, 22, 20, 1],
    ], columns=FEATURE_COLS)

    preds = model.predict(samples)
    labels = ["28d cycle/age25", "30d cycle/age30",
              "26d+bleeding/age28", "35d cycle/age22"]
    print(f"\n  {'Scenario':<30} {'Predicted Ovulation Day':>25}")
    print(f"  {'-'*55}")
    for label, pred in zip(labels, preds):
        formula = samples.iloc[labels.index(label)]["LengthofCycle"] - 14
        print(f"  {label:<30} Day {int(round(pred)):>3}  (formula: Day {int(formula)})")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load dataset
cervical = pd.read_csv(
    r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\final_dataset\cervical_cancer.csv'
)
# Clean column names
cervical.columns = cervical.columns.str.strip()
# Replace '?' with NaN
cervical = cervical.replace(r'\s*\?\s*', np.nan, regex=True)
# Convert all columns to numeric
for col in cervical.columns:
    cervical[col] = pd.to_numeric(cervical[col], errors='coerce')


# ---------------- PREPROCESSING ----------------
mean_impute_cols = [
    "Hormonal Contraceptives (years)",
]

unknown_class_cols = [
    "Number of sexual partners",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:HIV",
]

remove_cols = [
    "STDs:vaginal condylomatosis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
]

for col in mean_impute_cols:
    if col in cervical.columns:
        cervical[col] = cervical[col].fillna(cervical[col].mean())

for col in unknown_class_cols:
    if col in cervical.columns:
        cervical[col] = cervical[col].fillna(-1)

cervical = cervical.drop(columns=remove_cols, errors='ignore')


# ---------------- PROBABILITY TO RISK CATEGORY ----------------
def risk_category(prob):
    if prob >= 0.50:
        return "Very High Risk: Immediate clinical screening advised."
    elif prob >= 0.25:
        return "High Risk: Professional consultation recommended."
    elif prob >= 0.10:
        # Matches your 0.750 Recall threshold
        return "Moderate Risk: Elevated indicators; scheduling a Pap smear is advised."
    elif prob >= 0.05:
        return "Low Risk: Minor risk factors detected; maintain regular checkups."
    else:
        return "No Risk: No significant indicators found."

# ---------------- PRINT POSITIVE COUNTS ----------------
def print_positive_counts(y_train, y_valid, y_test):
    print("Train positives:", y_train.sum())
    print("Validation positives:", y_valid.sum())
    print("Test positives:", y_test.sum())

# ---------------- DATA SPLITTING ----------------
def cervical_data_split(data, test_size=0.3, random_state=42):
    X = data.drop('Biopsy', axis=1)
    y = data['Biopsy']

    x_train, x_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test

# ---------------- THRESHOLD TUNING FOR RECALL ----------------
def tune_threshold_for_recall(model, x_valid, y_valid):
    probs = model.predict_proba(x_valid)[:, 1]

    thresholds = np.arange(0.05, 0.51, 0.05)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_valid, preds).ravel()

        recall = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        results.append({
            "Threshold": t,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
            "False Positives": fp,
            "False Negatives": fn
        })

    df = pd.DataFrame(results)
    return df

# ---------------- LOGISTIC REGRESSION WITH CALIBRATION MODEL ----------------
def logistic_regression_with_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid",   # or "isotonic"
    threshold=0.10
):

    # Base pipeline (same as before)
    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])

    # Calibrated classifier
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv="prefit"   # we will fit manually
    )

    # Step 1: Train base model
    base_pipeline.fit(x_train, y_train)

    # Step 2: Calibrate on validation data
    calibrated_model.fit(x_valid, y_valid)

    # ---------------- VALIDATION METRICS ----------------
    y_valid_prob = calibrated_model.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= threshold).astype(int)

    print("\n--- CALIBRATED LOGISTIC REGRESSION MODEL METRICS ---")
    print("\nConfusion Matrix (Validation - Calibrated):")
    print(confusion_matrix(y_valid, y_valid_pred))

    print("\nClassification Report (Validation - Calibrated):")
    print(classification_report(y_valid, y_valid_pred, digits=3))

    roc_auc = roc_auc_score(y_valid, y_valid_prob)
    pr_auc = average_precision_score(y_valid, y_valid_prob)
    brier = brier_score_loss(y_valid, y_valid_prob)

    print(f"ROC-AUC (Validation): {roc_auc:.3f}")
    print(f"PR-AUC (Validation): {pr_auc:.3f}")
    print(f"Brier Score (Validation): {brier:.3f}")

    # Threshold tuning
    threshold_results = tune_threshold_for_recall(
        calibrated_model, x_valid, y_valid
    )
    print("\nThreshold Tuning Results:\n", threshold_results)

    chosen_threshold = 0.10

    print("\nChosen Threshold:", chosen_threshold)

    # Step 2: Calibrate on validation data
    calibrated_model.fit(x_valid, y_valid)

    # ---------------- TEST METRICS ----------------
    y_test_prob = calibrated_model.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_prob >= chosen_threshold).astype(int)

    print("\n--- CALIBRATED LOGISTIC REGRESSION MODEL METRICS ---")
    print("\nConfusion Matrix (Test - Calibrated):")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report (Test - Calibrated):")
    print(classification_report(y_test, y_test_pred, digits=3))

    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)
    brier = brier_score_loss(y_test, y_test_prob)

    print(f"ROC-AUC (Test): {roc_auc:.3f}")
    print(f"PR-AUC (Test): {pr_auc:.3f}")
    print(f"Brier Score (Test): {brier:.3f}")

    # ---------------- TEST SAMPLE RISK ----------------
    sample_patient = x_test.iloc[[0]]
    risk = calibrated_model.predict_proba(sample_patient)[0, 1]
    print(f"\nCalibrated cervical cancer risk (sample): {risk:.2%}")
    print(f"Applied decision threshold: {chosen_threshold:.2f}")

    return calibrated_model

# ----------------RUN DATA SPLITTING ----------------
# Perform split
x_train, x_valid, x_test, y_train, y_valid, y_test = cervical_data_split(cervical)
# Print positive counts
print_positive_counts(y_train, y_valid, y_test)

# ----------------RUN CALIBRATED LOGISTIC REGRESSION ----------------
calibrated_logistic_model = logistic_regression_with_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid",
    threshold=0.10
)

# ---------------- SAVE FINAL MODEL ----------------
import os

MODEL_DIR = r"C:\Users\user\SreelakshmiK\personal\Projects\She_Health_Clone\She-Health\backend\app\ml\models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "cervical_cancer_model.pkl")

# Save model
joblib.dump({
    "model": calibrated_logistic_model,
    "features": x_train.columns.tolist(),
    "threshold": 0.10
}, model_path)

print(f"\nModel saved successfully at: {model_path}")



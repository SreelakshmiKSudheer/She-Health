import numpy as np
import pandas as pd

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

# ---------------- PROBABILITY TO RISK CATEGORY ----------------
def risk_category(prob):
    if prob >= 0.70:
        return "Very High Risk"
    elif prob >= 0.50:
        return "High Risk"
    elif prob >= 0.30:
        return "Moderate Risk"
    elif prob >= 0.10:
        return "Low Risk"
    else:
        return "Very Low Risk"

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

# ---------------- LOGISTIC REGRESSION MODEL ----------------
def logistic_regression_model(x_train, y_train, x_valid, y_valid, x_test, y_test):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # 🔴 FIX 3
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train
    pipeline.fit(x_train, y_train)

    # Validation predictions
    y_valid_prob = pipeline.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- LOGISTIC REGRESSION MODEL METRICS ---")
    print("\nConfusion Matrix (Validation):")
    print(confusion_matrix(y_valid, y_valid_pred))

    print("\nClassification Report (Validation):")
    print(classification_report(y_valid, y_valid_pred, digits=3))

    roc_auc = roc_auc_score(y_valid, y_valid_prob)
    pr_auc = average_precision_score(y_valid, y_valid_prob)
    brier = brier_score_loss(y_valid, y_valid_prob)

    print(f"ROC-AUC (Validation): {roc_auc:.3f}")
    print(f"PR-AUC (Validation): {pr_auc:.3f}")
    print(f"Brier Score (Validation): {brier:.3f}")

    # Example risk prediction
    sample_patient = x_test.iloc[[0]]
    risk = pipeline.predict_proba(sample_patient)[0, 1]
    print(f"\nPredicted cervical cancer risk (sample): {risk:.2%}")
    print(f"Actual (sample): {y_test.iloc[0]}")

# ---------------- LOGISTIC REGRESSION WITH CALIBRATION MODEL ----------------
def logistic_regression_with_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"   # or "isotonic"
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

    # 🔴 Calibrated classifier
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
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

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

    # ---------------- TEST SAMPLE RISK ----------------
    sample_patient = x_test.iloc[[0]]
    risk = calibrated_model.predict_proba(sample_patient)[0, 1]
    print(f"\nCalibrated cervical cancer risk (sample): {risk:.2%}")

    return calibrated_model

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


# ---------------- RANDOM FOREST MODEL WITH TUNING & CALIBRATION ----------------
def random_forest_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
):

    # Base pipeline
    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
    ])

    # Hyperparameter grid
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )

    # Fit grid on training data
    grid.fit(x_train, y_train)

    best_pipeline = grid.best_estimator_
    print('\n--- RANDOM FOREST TUNING ---')
    print('Best params:', grid.best_params_)
    print('Best CV ROC-AUC:', grid.best_score_)

    # Calibrate using validation set (use prefit)
    calibrated = CalibratedClassifierCV(
        estimator=best_pipeline,
        method=method,
        cv='prefit'
    )

    calibrated.fit(x_valid, y_valid)

    # Validation metrics
    y_valid_prob = calibrated.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED RANDOM FOREST METRICS (Validation) ---")
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

    # Calibration curve values
    prob_true, prob_pred = calibration_curve(y_valid, y_valid_prob, n_bins=10)
    print('\nCalibration bins (pred, true):')
    for p_pred, p_true in zip(prob_pred, prob_true):
        print(f"Predicted: {p_pred:.2f}, True: {p_true:.2f}")

    # Threshold tuning for recall
    threshold_results = tune_threshold_for_recall(calibrated, x_valid, y_valid)
    print('\nThreshold tuning results (validation):')
    print(threshold_results)

    # Choose threshold achieving >=90% recall if available, otherwise default 0.5
    chosen_threshold = 0.5
    try:
        optimal_row = threshold_results[threshold_results["Recall"] >= 0.90].iloc[0]
        chosen_threshold = optimal_row["Threshold"]
    except Exception:
        # fallback keep 0.5
        pass

    print('\n--- TUNED CALIBRATED RANDOM FOREST METRICS ---')
    print('Chosen Threshold:', chosen_threshold)

    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    print('\nConfusion Matrix (Test):')
    print(confusion_matrix(y_test, test_preds))
    print('\nClassification Report (Test):')
    print(classification_report(y_test, test_preds, digits=3))
    print('ROC-AUC (Test):', roc_auc_score(y_test, test_probs))
    print('PR-AUC (Test):', average_precision_score(y_test, test_probs))
    print('Brier Score (Test):', brier_score_loss(y_test, test_probs))

    # Print sample risk categories
    risk_labels = [risk_category(p) for p in test_probs]
    print('\nRisk Categories (Test Samples):')
    for i in range(min(10, len(test_probs))):
        print(f"Sample {i+1}: Probability={test_probs[i]:.2%}, Category={risk_labels[i]}")

    return calibrated, grid


# ---------------- XGBOOST MODEL WITH TUNING & CALIBRATION ----------------
def xgboost_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
):

    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(use_label_encoder=False, random_state=42, n_jobs=-1, verbosity=0))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6, 10],
        'model__learning_rate': [0.01, 0.1],
        'model__subsample': [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(x_train, y_train)

    best_pipeline = grid.best_estimator_
    print('\n--- XGBOOST TUNING ---')
    print('Best params:', grid.best_params_)
    print('Best CV ROC-AUC:', grid.best_score_)

    calibrated = CalibratedClassifierCV(estimator=best_pipeline, method=method, cv='prefit')
    calibrated.fit(x_valid, y_valid)

    # Validation metrics
    y_valid_prob = calibrated.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED XGBOOST METRICS (Validation) ---")
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

    prob_true, prob_pred = calibration_curve(y_valid, y_valid_prob, n_bins=10)
    print('\nCalibration bins (pred, true):')
    for p_pred, p_true in zip(prob_pred, prob_true):
        print(f"Predicted: {p_pred:.2f}, True: {p_true:.2f}")

    threshold_results = tune_threshold_for_recall(calibrated, x_valid, y_valid)
    print('\nThreshold tuning results (validation):')
    print(threshold_results)

    chosen_threshold = 0.5
    try:
        optimal_row = threshold_results[threshold_results["Recall"] >= 0.90].iloc[0]
        chosen_threshold = optimal_row["Threshold"]
    except Exception:
        pass

    print('\n--- TUNED CALIBRATED XGBOOST METRICS ---')
    print('Chosen Threshold:', chosen_threshold)

    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    print('\nConfusion Matrix (Test):')
    print(confusion_matrix(y_test, test_preds))
    print('\nClassification Report (Test):')
    print(classification_report(y_test, test_preds, digits=3))
    print('ROC-AUC (Test):', roc_auc_score(y_test, test_probs))
    print('PR-AUC (Test):', average_precision_score(y_test, test_probs))
    print('Brier Score (Test):', brier_score_loss(y_test, test_probs))

    risk_labels = [risk_category(p) for p in test_probs]
    print('\nRisk Categories (Test Samples):')
    for i in range(min(10, len(test_probs))):
        print(f"Sample {i+1}: Probability={test_probs[i]:.2%}, Category={risk_labels[i]}")

    return calibrated, grid


# ---------------- LIGHTGBM MODEL WITH TUNING & CALIBRATION ----------------
def lightgbm_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
):

    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            objective='binary',
            class_weight='balanced',
            min_child_samples=5,
            min_split_gain=0.0,
            verbosity=-1
        ))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [-1, 6, 10],
        'model__learning_rate': [0.01, 0.1],
        'model__num_leaves': [31, 64],
        'model__subsample': [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(x_train, y_train)

    best_pipeline = grid.best_estimator_
    print('\n--- LIGHTGBM TUNING ---')
    print('Best params:', grid.best_params_)
    print('Best CV ROC-AUC:', grid.best_score_)

    calibrated = CalibratedClassifierCV(estimator=best_pipeline, method=method, cv='prefit')
    calibrated.fit(x_valid, y_valid)

    # Validation metrics
    y_valid_prob = calibrated.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED LIGHTGBM METRICS (Validation) ---")
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

    prob_true, prob_pred = calibration_curve(y_valid, y_valid_prob, n_bins=10)
    print('\nCalibration bins (pred, true):')
    for p_pred, p_true in zip(prob_pred, prob_true):
        print(f"Predicted: {p_pred:.2f}, True: {p_true:.2f}")

    threshold_results = tune_threshold_for_recall(calibrated, x_valid, y_valid)
    print('\nThreshold tuning results (validation):')
    print(threshold_results)

    chosen_threshold = 0.5
    try:
        optimal_row = threshold_results[threshold_results["Recall"] >= 0.90].iloc[0]
        chosen_threshold = optimal_row["Threshold"]
    except Exception:
        pass

    print('\n--- TUNED CALIBRATED LIGHTGBM METRICS ---')
    print('Chosen Threshold:', chosen_threshold)

    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    print('\nConfusion Matrix (Test):')
    print(confusion_matrix(y_test, test_preds))
    print('\nClassification Report (Test):')
    print(classification_report(y_test, test_preds, digits=3))
    print('ROC-AUC (Test):', roc_auc_score(y_test, test_probs))
    print('PR-AUC (Test):', average_precision_score(y_test, test_probs))
    print('Brier Score (Test):', brier_score_loss(y_test, test_probs))

    risk_labels = [risk_category(p) for p in test_probs]
    print('\nRisk Categories (Test Samples):')
    for i in range(min(10, len(test_probs))):
        print(f"Sample {i+1}: Probability={test_probs[i]:.2%}, Category={risk_labels[i]}")

    return calibrated, grid

# Perform split
x_train, x_valid, x_test, y_train, y_valid, y_test = cervical_data_split(cervical)
# Print positive counts
print_positive_counts(y_train, y_valid, y_test)

# ----------------RUN LOGISTIC REGRESSION MODEL ----------------
logistic_regression_model(x_train, y_train, x_valid, y_valid, x_test, y_test)

# ----------------RUN LOGISTIC REGRESSION WITH CALIBRATION MODEL ----------------
calibrated_model = logistic_regression_with_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"   # Platt Scaling
)
prob_true, prob_pred = calibration_curve(
    y_valid, 
    calibrated_model.predict_proba(x_valid)[:, 1],
    n_bins=10
)
for p_pred, p_true in zip(prob_pred, prob_true):
    print(f"Predicted: {p_pred:.2f}, True: {p_true:.2f}")

# ----------------TUNE LOGISTIC REGRESSION WITH CALIBRATION MODEL ----------------
threshold_results = tune_threshold_for_recall(
    calibrated_model,
    x_valid,
    y_valid
)
print(threshold_results)
optimal_row = threshold_results[threshold_results["Recall"] >= 0.90].iloc[0]
optimal_threshold = optimal_row["Threshold"]

print("\n--- TUNED CALIBRATED LOGISTIC REGRESSION MODEL METRICS ---")
print("Chosen Threshold:", optimal_threshold)

test_probs = calibrated_model.predict_proba(x_test)[:, 1]
test_preds = (test_probs >= optimal_threshold).astype(int)

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, test_preds))
print("\nClassification Report (Test):")
print(classification_report(y_test, test_preds, digits=3))
print("ROC-AUC (Test):", roc_auc_score(y_test, test_probs))
print("PR-AUC (Test):", average_precision_score(y_test, test_probs))
print("Brier Score (Test):", brier_score_loss(y_test, test_probs))


risk_labels = [risk_category(p) for p in test_probs]
print("\nRisk Categories (Test Samples):")
for i in range(10):
    print(f"Sample {i+1}: Probability={test_probs[i]:.2%}, Category={risk_labels[i]}")


# ----------------RUN RANDOM FOREST WITH TUNING & CALIBRATION ----------------
calibrated_rf, rf_grid = random_forest_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
)


# ----------------RUN XGBOOST WITH TUNING & CALIBRATION ----------------
calibrated_xgb, xgb_grid = xgboost_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
)


# ----------------RUN LIGHTGBM WITH TUNING & CALIBRATION ----------------
calibrated_lgbm, lgbm_grid = lightgbm_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
)

# FINAL MODEL SELECTION CAN BE DONE BASED ON THE METRICS PRINTED ABOVE
'''
Among all evaluated models, the tuned calibrated logistic regression was selected as the final model due to its superior sensitivity and lower false-negative rate on the test set. While ensemble and boosting models demonstrated competitive discrimination performance, they exhibited reduced recall, making them less suitable for screening-oriented cervical cancer risk prediction.
'''
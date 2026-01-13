import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# PREPROCESSING PIPELINE IMPORTS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# MODEL IMPORTS
from sklearn.linear_model import LogisticRegression
# EVALUATION METRICS IMPORTS
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
# CALIBRATION IMPORT
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------- PROBABILITY TO RISK CATEGORY ----------------
def risk_category(prob):
    if prob >= 0.70:
        return "Very High Risk"
    elif prob >= 0.40:
        return "High Risk"
    elif prob >= 0.20:
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
def data_split(data, test_size=0.3, random_state=42):
    X = data.drop('label', axis=1)
    y = data['label']

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
def logistic_regression(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test
    
):

    pipeline = Pipeline([

        ("model", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])

    pipeline.fit(x_train, y_train)

    y_valid_prob = pipeline.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- PCOS LOGISTIC REGRESSION METRICS ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))
    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

    sample = x_test.iloc[[0]]
    risk = pipeline.predict_proba(sample)[0, 1]
    print(f"\nPredicted PCOS risk (sample): {risk:.2%}")
    print(f"Actual (sample): {y_test.iloc[0]}")

    return pipeline

# ---------------- LOGISTIC REGRESSION WITH CALIBRATION MODEL ----------------
def logistic_with_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
):

    base_pipeline = Pipeline([
        ("model", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train base model
    base_pipeline.fit(x_train, y_train)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv="prefit"
    )

    calibrated_model.fit(x_valid, y_valid)

    y_valid_prob = calibrated_model.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED PCOS LOGISTIC REGRESSION METRICS ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))
    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

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

# ---------------- ENDOMETRIOSIS RANDOM FOREST WITH TUNING AND CALIBRATION ----------------
def endometriosis_random_forest_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
):
    """
    Trains a Random Forest for Endometriosis risk prediction,
    performs hyperparameter tuning, calibrates probabilities,
    tunes threshold for high recall, evaluates, and prints metrics.
    """

    # Identify feature types
    numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x_train.select_dtypes(include=["object", "category"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    # Pipeline
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Hyperparameters
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    # Train
    grid.fit(x_train, y_train)
    best_pipeline = grid.best_estimator_

    print("\n--- ENDOMETRIOSIS RANDOM FOREST TUNING ---")
    print("Best params:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)

    # Calibration
    calibrated = CalibratedClassifierCV(
        estimator=best_pipeline,
        method=method,
        cv="prefit"
    )
    calibrated.fit(x_valid, y_valid)

    # Validation metrics
    y_valid_prob = calibrated.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED ENDOMETRIOSIS RANDOM FOREST (Validation) ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))
    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

    # Threshold tuning (recall ≥ 80%)
    threshold_results = tune_threshold_for_recall(calibrated, x_valid, y_valid)

    chosen_threshold = 0.5
    try:
        chosen_threshold = threshold_results[
            threshold_results["Recall"] >= 0.80
        ].iloc[0]["Threshold"]
    except Exception:
        pass

    print("\nChosen Threshold:", chosen_threshold)

    # Test metrics
    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_preds))
    print(classification_report(y_test, test_preds, digits=3))
    print("ROC-AUC (Test):", roc_auc_score(y_test, test_probs))
    print("PR-AUC (Test):", average_precision_score(y_test, test_probs))
    print("Brier Score (Test):", brier_score_loss(y_test, test_probs))

    # Risk categories
    print("\nEndometriosis Risk Categories (Samples):")
    for i in range(min(10, len(test_probs))):
        print(
            f"Sample {i+1}: "
            f"Probability={test_probs[i]:.2%}, "
            f"Category={risk_category(test_probs[i])}"
        )

    return calibrated, grid

# ---------------- XGBOOST MODEL WITH TUNING & CALIBRATION FOR ENDOMETRIOSIS ----------------
def endometriosis_xgboost_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
):

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Identify feature types
    numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x_train.select_dtypes(include=["object", "category"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    # Pipeline
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ])

    # Hyperparameter grid
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 6, 10],
        "model__learning_rate": [0.01, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    # Train
    grid.fit(x_train, y_train)
    best_pipeline = grid.best_estimator_

    print("\n--- ENDOMETRIOSIS XGBOOST TUNING ---")
    print("Best params:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)

    # Calibration
    calibrated = CalibratedClassifierCV(
        estimator=best_pipeline,
        method=method,
        cv="prefit"
    )
    calibrated.fit(x_valid, y_valid)

    # Validation
    y_valid_prob = calibrated.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED ENDOMETRIOSIS XGBOOST (Validation) ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))
    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

    # Threshold tuning (recall ≥ 80%)
    threshold_results = tune_threshold_for_recall(
        calibrated, x_valid, y_valid
    )

    chosen_threshold = 0.5
    try:
        chosen_threshold = threshold_results[
            threshold_results["Recall"] >= 0.80
        ].iloc[0]["Threshold"]
    except Exception:
        pass

    print("\nChosen Threshold:", chosen_threshold)

    # Test
    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_preds))
    print(classification_report(y_test, test_preds, digits=3))
    print("ROC-AUC (Test):", roc_auc_score(y_test, test_probs))
    print("PR-AUC (Test):", average_precision_score(y_test, test_probs))
    print("Brier Score (Test):", brier_score_loss(y_test, test_probs))

    print("\nEndometriosis Risk Categories (Samples):")
    for i in range(min(10, len(test_probs))):
        print(
            f"Sample {i+1}: "
            f"Probability={test_probs[i]:.2%}, "
            f"Category={risk_category(test_probs[i])}"
        )

    return calibrated, grid

# ---------------- LIGHTGBM ENDOMETRIOSIS MODEL WITH TUNING & CALIBRATION ----------------
def lightgbm_endometriosis_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
):

    # Identify feature types
    numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x_train.select_dtypes(include=["object", "category"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    # Pipeline (NO scaling needed for LightGBM)
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            objective="binary",
            class_weight="balanced",
            min_child_samples=5,
            min_split_gain=0.0,
            verbosity=-1
        ))
    ])

    # Hyperparameter grid
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [-1, 6, 10],
        "model__learning_rate": [0.01, 0.1],
        "model__num_leaves": [31, 64],
        "model__subsample": [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    # Train
    grid.fit(x_train, y_train)
    best_pipeline = grid.best_estimator_

    print("\n--- LIGHTGBM ENDOMETRIOSIS TUNING ---")
    print("Best params:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)

    # Calibration
    calibrated = CalibratedClassifierCV(
        estimator=best_pipeline,
        method=method,
        cv="prefit"
    )
    calibrated.fit(x_valid, y_valid)

    # Validation
    y_valid_prob = calibrated.predict_proba(x_valid)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED LIGHTGBM ENDOMETRIOSIS (Validation) ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))

    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

    # Threshold tuning (Endometriosis → recall ≥ 90%)
    threshold_results = tune_threshold_for_recall(
        calibrated, x_valid, y_valid
    )

    chosen_threshold = 0.5
    try:
        chosen_threshold = threshold_results[
            threshold_results["Recall"] >= 0.90
        ].iloc[0]["Threshold"]
    except:
        pass

    print("Chosen Threshold:", chosen_threshold)

    # Test
    test_probs = calibrated.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_preds))
    print(classification_report(y_test, test_preds, digits=3))

    print("ROC-AUC (Test):", roc_auc_score(y_test, test_probs))
    print("PR-AUC (Test):", average_precision_score(y_test, test_probs))
    print("Brier Score (Test):", brier_score_loss(y_test, test_probs))

    # Risk categories
    print("\nEndometriosis Risk Categories (Samples):")
    for i in range(min(10, len(test_probs))):
        print(
            f"Sample {i+1}: "
            f"Probability={test_probs[i]:.2%}, "
            f"Risk={risk_category(test_probs[i])}"
        )

    return calibrated, grid


# Load cleaned data
endometriosis = pd.read_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She-Health\backend\dataset\final_dataset\endometriosis.csv')
endometriosis.columns = endometriosis.columns.str.strip()

# Initial data inspection
'''
print(endometriosis.shape)
print(endometriosis['label'].value_counts())
print(endometriosis.dtypes)
print("\nMissing values after cleaning:\n", endometriosis.isnull().sum())

print(endometriosis["label"].unique())
print(endometriosis["label"].value_counts(dropna=False))
'''
X = endometriosis.drop(columns=["label"])
y = endometriosis["label"]

x_train, x_valid, x_test, y_train, y_valid, y_test = data_split(endometriosis)
print_positive_counts(y_train, y_valid, y_test)

# ---------------- RUN LOGISTIC REGRESSION AND CALIBRATED MODEL ----------------

logistic_regression(x_train, y_train, x_valid, y_valid, x_test, y_test)
# ----------------- LOGISTIC REGRESSION WITH CALIBRATION ----------------
calibrated_model = logistic_with_calibration(x_train, y_train, x_valid, y_valid, x_test, y_test, method="sigmoid")

threshold_results = tune_threshold_for_recall(
    calibrated_model,
    x_valid,
    y_valid
)
print(threshold_results)
optimal_threshold = threshold_results[
    threshold_results["Recall"] >= 0.90
].iloc[0]["Threshold"]


test_probs = calibrated_model.predict_proba(x_test)[:, 1]
test_preds = (test_probs >= optimal_threshold).astype(int)

print("\n--- TUNED CALIBRATED ENDOMETRIOSIS MODEL METRICS ---")
print("Chosen Threshold:", optimal_threshold)
print(confusion_matrix(y_test, test_preds))
print(classification_report(y_test, test_preds, digits=3))
print("ROC-AUC:", roc_auc_score(y_test, test_probs))
print("PR-AUC:", average_precision_score(y_test, test_probs))
print("Brier Score:", brier_score_loss(y_test, test_probs))

risk_labels = [risk_category(p) for p in test_probs[:10]]
for i, (p, r) in enumerate(zip(test_probs[:10], risk_labels)):
    print(f"Sample {i+1}: Probability={p:.2%}, Category={r}")


# ----------------RUN RANDOM FOREST WITH TUNING & CALIBRATION ----------------
calibrated_rf, rf_grid = endometriosis_random_forest_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
)

# ----------------RUN XGBOOST WITH TUNING & CALIBRATION ----------------
calibrated_xgb, xgb_grid = endometriosis_xgboost_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
)


# ----------------RUN LIGHTGBM WITH TUNING & CALIBRATION ----------------
calibrated_lgbm, lgbm_grid = lightgbm_endometriosis_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
)
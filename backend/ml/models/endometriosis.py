import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# PREPROCESSING PIPELINE IMPORTS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# MODEL IMPORTS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

# ---------------- PROBABILITY TO RISK CATEGORY ----------------
def risk_category(prob):
    if prob >= 0.70:
        return "Very High Risk"
    elif prob >= 0.50:
        return "High Risk"
    elif prob >= 0.25:
        return "Moderate Risk"
    elif prob >= 0.10:
        return "Low Risk"
    else:
        return "No Risk"

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

# -------- 24 FEATURES FOR ADABOOST MODEL --------
adaboost_features = [
    'Heavy / Extreme menstrual bleeding',
    'Irregular / Missed periods',
    'Abnormal uterine bleeding',
    'Menstrual pain (Dysmenorrhea)',
    'Painful bowel movements',
    'Bowel pain',
    'Pelvic pain',
    'IBS-like symptoms',
    'Painful cramps during period',
    'Fatigue / Chronic fatigue',
    'Loss of appetite',
    'Constant bleeding',
    'Painful ovulation',
    'Hormonal problems',
    'Malaise / Sickness',
    'Fever',
    'Cramping',
    'Bloating',
    'Painful / Burning pain during sex (Dyspareunia)',
    'Extreme / Severe pain',
    'Pain / Chronic pain',
    'Ovarian cysts',
    'Fertility Issues',
    'Feeling sick'
]

X = endometriosis[adaboost_features]
y = endometriosis["label"]

x_train, x_valid, x_test, y_train, y_valid, y_test = data_split(endometriosis)
print_positive_counts(y_train, y_valid, y_test)


# -------- THRESHOLD TUNING FOR RECALL --------
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


# -------- ADABOOST MODEL WITH TUNING & CALIBRATION --------
def endometriosis_adaboost_with_tuning_and_calibration(
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
        ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=42
        ))
    ])

    # Hyperparameter grid
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.5, 1.0, 1.5],
        "model__estimator__max_depth": [1, 2, 3]
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

    print("\n--- ENDOMETRIOSIS ADABOOST TUNING ---")
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

    print("\n--- CALIBRATED ENDOMETRIOSIS ADABOOST (Validation) ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))

    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

    # Threshold tuning
    threshold_results = tune_threshold_for_recall(
        calibrated, x_valid, y_valid
    )
    print("\nThreshold Tuning Results:\n", threshold_results)

    chosen_threshold = 0.35

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

    # ---- PRINT FEATURE IMPORTANCE (UNCALIBRATED) ----
    print("\n--- FEATURE IMPORTANCE (AdaBoost) ---")
    adaboost_model_uncal = best_pipeline.named_steps["model"]
    feature_importance_uncal = adaboost_model_uncal.feature_importances_
    
    # Get feature names - all selected features are numeric
    feature_names = list(x_train.columns)
    
    # Create feature importance dataframe
    feature_imp_df_uncal = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance_uncal
    }).sort_values("Importance", ascending=False)
    
    print("\nFeature Importance Scores:")
    print(feature_imp_df_uncal.to_string(index=False))
    print(f"\nTop 10 Most Important Features:")
    print(feature_imp_df_uncal.head(10).to_string(index=False))

    return calibrated, grid


# -------- PREPARE DATA FOR ADABOOST --------
x_train_ada = x_train[adaboost_features]
x_valid_ada = x_valid[adaboost_features]
x_test_ada = x_test[adaboost_features]

# -------- RUN ADABOOST WITH TUNING & CALIBRATION --------
calibrated_ada, ada_grid = endometriosis_adaboost_with_tuning_and_calibration(
    x_train_ada, y_train,
    x_valid_ada, y_valid,
    x_test_ada, y_test,
    method="sigmoid"
)


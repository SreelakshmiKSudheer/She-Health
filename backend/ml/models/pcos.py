# DATA MANIPULATION IMPORTS
import pandas as pd
import numpy as np
# DATA SPLITTING & MODEL SELECTION IMPORTS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# PREPROCESSING PIPELINE IMPORTS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
# MODEL IMPORTS
from xgboost import XGBClassifier

# FEATURE SELECTION IMPORTS
# Filter Method (Fast Screening) - Remove low-importance features
from sklearn.feature_selection import VarianceThreshold
# Stage B: Statistical Selection - Remove features with low correlation to target
from sklearn.feature_selection import SelectKBest, f_classif
# Stage C: Wrapper Method (Refinement) - Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# Load cleaned data
pcos = pd.read_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She_Health_Clone\She-Health\backend\dataset\final_dataset\pcos.csv')
pcos.columns = pcos.columns.str.strip()

# Drop specified columns
pcos = pcos.drop(columns=["City", 'loss weight gain / weight loss', 'more Mood Swings'], errors="ignore")

'''
# Initial data inspection
print(pcos.shape)
print(pcos['PCOS'].value_counts())
print(pcos.dtypes)
print("\nMissing values after cleaning:\n", pcos.isnull().sum())
'''

X = pcos.drop(columns=["PCOS"])
y = pcos["PCOS"]

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
def pcos_data_split(data, test_size=0.4, random_state=42):
    X = data.drop('PCOS', axis=1)
    y = data['PCOS']

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

def get_selected_feature_names(pipeline, numeric_features, categorical_features):
    # Step 1: Get preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]

    # Step 2: Get feature names after preprocessing
    num_features = numeric_features.tolist()

    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_features = cat_encoder.get_feature_names_out(categorical_features)

    all_features = np.concatenate([num_features, cat_features])

    # Step 3: Apply VarianceThreshold mask
    var_mask = pipeline.named_steps["feature_selection"]\
        .named_steps["variance"].get_support()
    features_after_var = all_features[var_mask]

    # Step 4: Apply SelectKBest mask
    kbest_mask = pipeline.named_steps["feature_selection"]\
        .named_steps["select_kbest"].get_support()
    features_after_kbest = features_after_var[kbest_mask]

    # Step 5: Apply RFE mask
    rfe_mask = pipeline.named_steps["feature_selection"]\
        .named_steps["rfe"].get_support()
    final_features = features_after_kbest[rfe_mask]

    return final_features

# ---------------- XGBOOST MODEL WITH TUNING & CALIBRATION ----------------
def pcos_xgboost_with_tuning_and_calibration(
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

    # FEATURE SELECTION PIPELINE
    feature_selector = Pipeline([
        # Stage A: Remove low variance features
        ("variance", VarianceThreshold(threshold=0.01)),

        # Stage B: Select top features statistically
        ("select_kbest", SelectKBest(score_func=f_classif, k=25)),  # adjust k if needed
        
        # Stage C: Optimized RFE (wrapper method)
        ("rfe", RFE(
            estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            n_features_to_select=18,   # final feature count
            step=2                     # removes 2 features per iteration (faster)
        ))
    ])

    # Pipeline
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selector),
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
        # Feature Selection tuning
        "feature_selection__select_kbest__k": [25, 30, 35],
        "feature_selection__rfe__n_features_to_select": [18, 20, 22],

        # XGBoost tuning
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

    # -------- PRINT SELECTED FEATURES --------
    selected_features = get_selected_feature_names(
        best_pipeline,
        numeric_features,
        categorical_features
    )

    print("\n--- FINAL SELECTED FEATURES ---")
    for f in selected_features:
        print(f)

    print("\nTotal Selected Features:", len(selected_features))

    print("\n--- FEATURE IMPORTANCE (XGBoost) ---")
    model = best_pipeline.named_steps["model"]

    importances = model.feature_importances_

    for f, imp in zip(selected_features, importances):
        print(f"{f}: {imp:.4f}")

    print("\n--- PCOS XGBOOST TUNING ---")
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

    print("\n--- CALIBRATED PCOS XGBOOST (Validation) ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))

    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

    # Threshold tuning (PCOS → recall ≥ 80%)
    threshold_results = tune_threshold_for_recall(
        calibrated, x_valid, y_valid
    )
    print("\nThreshold Tuning Results:\n", threshold_results)
    '''
    chosen_threshold = 0.5
    try:
        chosen_threshold = threshold_results[
            threshold_results["Recall"] >= 0.80
        ].iloc[0]["Threshold"]
    except Exception:
        pass
    '''
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

    print("\nPCOS Risk Categories (Samples):")
    for i in range(min(10, len(test_probs))):
        print(
            f"Sample {i+1}: "
            f"Probability={test_probs[i]:.2%}, "
            f"Category={risk_category(test_probs[i])}"
        )
    return calibrated, grid

pcos["Overweight"] = pcos["Overweight"].fillna("Unknown")
x_train, x_valid, x_test, y_train, y_valid, y_test = pcos_data_split(pcos)
print_positive_counts(y_train, y_valid, y_test)

binary_features = [ 'Overweight',  
         'irregular or missed periods', 
         'Acne or skin tags', 
         'Hair thinning or hair loss', 
         'Dark patches', 
         'always tired', 
         'canned food often']

ternary_features = ['Hair growth  on Cheeks', 
           'Hair growth Between breasts',
           'Hair growth  on Upper lips',
           'Hair growth in Arms',
           'Hair growth on Inner thighs']

numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = x_train.select_dtypes(include=["object", "category"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ----------------RUN XGBOOST WITH TUNING & CALIBRATION ----------------
calibrated_xgb, xgb_grid = pcos_xgboost_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid"
)



# ---------------- SAVE FINAL MODEL ----------------
# import os
# import joblib

# MODEL_DIR = r"C:\Users\user\SreelakshmiK\personal\Projects\She_Health_Clone\She-Health\backend\app\ml\models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# model_path = os.path.join(MODEL_DIR, "pcos_model.pkl")

# # Save model
# joblib.dump({
#     "model": calibrated_xgb,
#     "features": x_train.columns.tolist(),
#     "threshold": 0.10  # your chosen threshold
# }, model_path)

# print(f"\nModel saved successfully at: {model_path}")
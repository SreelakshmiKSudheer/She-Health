import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# PREPROCESSING PIPELINE IMPORTS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# FEATURE SELECTION IMPORTS
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.base import BaseEstimator, TransformerMixin
# MODEL IMPORTS
from sklearn.linear_model import LogisticRegression
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
from sklearn.calibration import CalibratedClassifierCV

import os
import joblib


class SafeSelectKBest(SelectKBest):
    def fit(self, X, y):
        if self.k != "all":
            self.k = int(max(1, min(self.k, X.shape[1])))
        return super().fit(X, y)


class SafeRFE(RFE):
    def fit(self, X, y, **fit_params):
        if self.n_features_to_select is not None:
            self.n_features_to_select = int(max(1, min(self.n_features_to_select, X.shape[1])))
        return super().fit(X, y, **fit_params)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        if hasattr(X, "shape"):
            df = pd.DataFrame(X)
        else:
            df = pd.DataFrame(X)

        # compute absolute correlation matrix and mark features to keep
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.support_ = np.array([col not in to_drop for col in df.columns], dtype=bool)
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.loc[:, self.support_].values

    def get_support(self):
        return self.support_


# ============ UTILITY FUNCTIONS ============

def risk_category(prob):
    """Categorize thyroid disease risk based on probability."""
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


def print_positive_counts(y_train, y_valid, y_test):
    """Print positive class counts in train/valid/test splits."""
    print("Train positives:", y_train.sum())
    print("Validation positives:", y_valid.sum())
    print("Test positives:", y_test.sum())


# ============ DATA SPLITTING ============

def data_split(data, test_size=0.3, random_state=42):
    """Split data into train, validation, and test sets with stratification."""
    X = data.drop('binaryClass', axis=1)
    y = data['binaryClass']

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


# ============ PREPROCESSING ============

def fit_preprocessor(x_train):
    """Fit a ColumnTransformer on training data and return the fitted preprocessor
    and list of output feature names."""
    numeric_features = x_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x_train.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    preprocessor.fit(x_train)

    # Build feature names
    num_names = list(numeric_features)
    cat_names = []
    if len(categorical_features) > 0:
        try:
            cat_transformer = preprocessor.named_transformers_.get("cat")
            encoder = None
            if hasattr(cat_transformer, "named_steps"):
                encoder = cat_transformer.named_steps.get("encoder")
            else:
                encoder = cat_transformer
            if encoder is not None and hasattr(encoder, "get_feature_names_out"):
                cat_names = list(encoder.get_feature_names_out(categorical_features))
        except Exception:
            cat_names = []

    feature_names = num_names + cat_names
    return preprocessor, feature_names


def transform_with_preprocessor(preprocessor, X, feature_names=None):
    """Transform X using a fitted preprocessor and return a DataFrame with
    `feature_names` as columns when available."""
    arr = preprocessor.transform(X)
    if feature_names is None:
        return pd.DataFrame(arr, index=X.index)
    return pd.DataFrame(arr, columns=feature_names, index=X.index)


def prepare_thyroid_model_data(x_train, x_valid, x_test):
    """Fit preprocessing once and return dense matrices plus feature metadata."""
    preprocessor, feature_names = fit_preprocessor(x_train)
    x_train_df = transform_with_preprocessor(preprocessor, x_train, feature_names)
    x_valid_df = transform_with_preprocessor(preprocessor, x_valid, feature_names)
    x_test_df = transform_with_preprocessor(preprocessor, x_test, feature_names)
    return (
        preprocessor,
        feature_names,
        x_train_df.values,
        x_valid_df.values,
        x_test_df.values,
    )


# ============ FEATURE SELECTION PIPELINE ============

def get_feature_selection_pipeline(n_features):
    """Return a feature selection Pipeline and a safe param grid for tuning.

    The selection grid is clipped to the actual number of features so grid search
    never asks SelectKBest or RFE for an impossible value.
    """

    n_features = int(max(1, n_features))

    k_candidates = sorted({min(n_features, value) for value in (17, 19, 21, n_features)})
    rfe_candidates = sorted({min(n_features, value) for value in (16, 18, 20, n_features)})

    valid_selection_pairs = []
    for k_value in k_candidates:
        valid_rfe_values = [rfe_value for rfe_value in rfe_candidates if rfe_value <= k_value]
        if not valid_rfe_values:
            valid_rfe_values = [k_value]
        for rfe_value in valid_rfe_values:
            valid_selection_pairs.append((k_value, rfe_value))

    feature_selector = Pipeline([
        ("variance", VarianceThreshold(threshold=0.01)),
        ("corr", CorrelationFilter(threshold=0.9)),
        ("select_kbest", SafeSelectKBest(score_func=f_classif, k=min(19, n_features))),
        ("rfe", SafeRFE(
            estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            n_features_to_select=min(18, n_features),
            step=2
        ))
    ])

    selection_param_grid = [
        {
            "feature_selection__select_kbest__k": [k_value],
            "feature_selection__rfe__n_features_to_select": [rfe_value],
        }
        for k_value, rfe_value in valid_selection_pairs
    ]

    return feature_selector, selection_param_grid


def get_selected_feature_names(feature_selection_pipeline, original_feature_names):
    """Compute final selected feature names after fitting the feature selection pipeline.

    This implementation is robust to transformers that change column counts: it
    propagates an identity matrix through each transform to track which original
    feature(s) contribute to each intermediate column, and applies any
    get_support() masks when present.
    """
    try:
        orig_names = list(original_feature_names)
        if len(orig_names) == 0:
            return []

        # mapping: list where mapping[i] is list of original feature names that
        # contribute to the i-th column at the current pipeline stage.
        mapping = [[n] for n in orig_names]
        prev_n = len(mapping)

        for step_name, step in feature_selection_pipeline.named_steps.items():
            # first compute how the step maps previous columns to next columns
            try:
                mat = step.transform(np.eye(prev_n))
            except Exception:
                # some transformers may not accept an identity; try a float identity
                mat = step.transform(np.eye(prev_n, dtype=float))

            if mat.ndim == 1:
                mat = mat.reshape(-1, 1)

            n_next = mat.shape[1]
            new_mapping = []
            for j in range(n_next):
                contrib = np.where(np.abs(mat[:, j]) > 1e-12)[0]
                mapped = []
                for idx in contrib:
                    mapped.extend(mapping[idx])
                new_mapping.append(list(dict.fromkeys(mapped)))

            mapping = new_mapping
            prev_n = len(mapping)

            # if the transformer exposes a get_support mask, apply it to mapping
            if hasattr(step, "get_support"):
                mask = np.asarray(step.get_support())
                if mask.dtype == bool:
                    mask_bool = mask
                else:
                    mask_bool = mask.astype(bool)
                if len(mask_bool) == len(mapping):
                    mapping = [m for m, keep in zip(mapping, mask_bool) if keep]
                    prev_n = len(mapping)

        # At the end, mapping contains lists of original feature names for each
        # final column; the selected original features are the union of those lists.
        selected = []
        for group in mapping:
            for name in group:
                if name not in selected:
                    selected.append(name)
        return selected
    except Exception:
        return None


def print_selected_features(feature_selection_pipeline, original_feature_names, title="Selected Features"):
    """Print selected feature names from a fitted feature selection pipeline."""
    selected_feature_names = get_selected_feature_names(feature_selection_pipeline, original_feature_names)
    if selected_feature_names is None:
        return None

    print(f"\n--- {title} ---")
    print(f"Count: {len(selected_feature_names)}")
    for feature_name in selected_feature_names:
        print(feature_name)

    return selected_feature_names


def print_final_selected_features(model_name, fitted_pipeline, original_feature_names):
    """Print the final selected feature names for a fitted thyroid model pipeline."""
    return print_selected_features(
        fitted_pipeline.named_steps.get("feature_selection"),
        original_feature_names,
        title=f"FINAL SELECTED FEATURES - {model_name}"
    )


# ============ THRESHOLD TUNING ============

def tune_threshold_for_recall(model, x_valid, y_valid):
    """Tune decision threshold to maximize recall with high precision."""
    probs = model.predict_proba(x_valid)[:, 1]

    thresholds = np.arange(0.05, 0.51, 0.05)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_valid, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0]) if cm.shape[0] == 1 else (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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


# ============ ADABOOST MODEL FOR THYROID PREDICTION ============

def thyroid_adaboost_with_tuning_and_calibration(
    x_train, y_train,
    x_valid, y_valid,
    x_test, y_test,
    method="sigmoid",
    feature_names=None
):
    """
    Trains an AdaBoost model for Thyroid risk prediction,
    performs hyperparameter tuning, calibrates probabilities,
    tunes threshold for high recall, evaluates, and prints metrics.
    """

    # Full flow: Preprocessing -> Filter -> SelectKBest -> RFE -> Train
    preprocessor, feature_names, x_train_p, x_valid_p, x_test_p = prepare_thyroid_model_data(
        x_train, x_valid, x_test
    )

    # Build feature selection + estimator pipeline
    feature_selector, selection_param_grid = get_feature_selection_pipeline(x_train_p.shape[1])
    base_pipeline = Pipeline([
        ("feature_selection", feature_selector),
        ("model", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=42
        ))
    ])

    # Hyperparameter grid (merge selection tuning with model tuning)
    model_param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.5, 1.0, 1.5],
        "model__estimator__max_depth": [1, 2, 3]
    }
    param_grid = [{**selection_grid, **model_param_grid} for selection_grid in selection_param_grid]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    # Train
    grid.fit(x_train_p, y_train)
    best_pipeline = grid.best_estimator_

    print("\n--- THYROID ADABOOST TUNING ---")
    print("Best params:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)
    print_final_selected_features("AdaBoost", best_pipeline, feature_names)

    # Print feature importance
    print("\n--- FEATURE IMPORTANCE (AdaBoost) ---")
    model = best_pipeline.named_steps.get("model")
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        for i, imp in enumerate(importances):
            print(f"f{i}: {imp:.4f}")

    # Calibration
    calibrated = CalibratedClassifierCV(
        estimator=best_pipeline,
        method=method,
        cv="prefit"
    )
    calibrated.fit(x_valid_p, y_valid)

    # Validation
    y_valid_prob = calibrated.predict_proba(x_valid_p)[:, 1]
    y_valid_pred = (y_valid_prob >= 0.5).astype(int)

    print("\n--- CALIBRATED THYROID ADABOOST (Validation) ---")
    print(confusion_matrix(y_valid, y_valid_pred))
    print(classification_report(y_valid, y_valid_pred, digits=3))

    print("ROC-AUC:", roc_auc_score(y_valid, y_valid_prob))
    print("PR-AUC:", average_precision_score(y_valid, y_valid_prob))
    print("Brier Score:", brier_score_loss(y_valid, y_valid_prob))

    # Threshold tuning
    threshold_results = tune_threshold_for_recall(
        calibrated, x_valid_p, y_valid
    )
    print("\nThreshold Tuning Results:\n", threshold_results)

    chosen_threshold = 0.40

    print("\nChosen Threshold:", chosen_threshold)

    # Test
    test_probs = calibrated.predict_proba(x_test_p)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_preds))
    print(classification_report(y_test, test_preds, digits=3))

    print("ROC-AUC (Test):", roc_auc_score(y_test, test_probs))
    print("PR-AUC (Test):", average_precision_score(y_test, test_probs))
    print("Brier Score (Test):", brier_score_loss(y_test, test_probs))

    print("\nThyroid Risk Categories (Samples):")
    for i in range(min(10, len(test_probs))):
        print(
            f"Sample {i+1}: "
            f"Probability={test_probs[i]:.2%}, "
            f"Category={risk_category(test_probs[i])}"
        )

    return calibrated, grid



if __name__ == "__main__":
    thyroid = pd.read_csv(r'C:\Users\user\SreelakshmiK\personal\Projects\She_Health_Clone\She-Health\backend\dataset\final_dataset\thyroid.csv')
    thyroid.columns = thyroid.columns.str.strip()

    # Initial data inspection
    '''
    print(thyroid.shape)
    print(thyroid['label'].value_counts())
    print(thyroid.dtypes)
    print("\nMissing values after cleaning:\n", thyroid.isnull().sum())

    print(thyroid["label"].unique())
    print(thyroid["label"].value_counts(dropna=False))
    '''
    X = thyroid.drop(columns=["binaryClass"])
    y = thyroid["binaryClass"]

    x_train, x_valid, x_test, y_train, y_valid, y_test = data_split(thyroid)
    print_positive_counts(y_train, y_valid, y_test)

    # AdaBoost with tuning & calibration
    calibrated_ada, ada_grid = thyroid_adaboost_with_tuning_and_calibration(
        x_train, y_train,
        x_valid, y_valid,
        x_test, y_test,
        method="sigmoid"
    )

    # ---------------- SAVE FINAL MODEL ----------------
    

    MODEL_DIR = r"C:\Users\user\SreelakshmiK\personal\Projects\She_Health_Clone\She-Health\backend\app\ml\models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "thyroid_model.pkl")

    # Save model
    joblib.dump({
        "model": calibrated_ada,
        "features": x_train.columns.tolist(),
        "threshold": 0.40  # your chosen threshold
    }, model_path)

    print(f"\nModel saved successfully at: {model_path}")
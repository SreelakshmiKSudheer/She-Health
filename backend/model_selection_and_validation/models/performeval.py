"""Model training utilities for health datasets.

This script contains helpers to train classifiers on the project's
datasets (thyroid, cervical, PCOS, endometriosis). It provides a
consistent workflow: preprocessing, train/test split, fitting,
metric computation, saving predictions and pickled models, and
logging per-run metrics to a summary text file under an
`outputs/` folder next to this file.
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from typing import Any, Dict, Optional
import os
from datetime import datetime

# Optional imports for additional model families. If a package isn't
# available the script will skip that model and continue.
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Read datasets used by this helper. Paths are relative to this script's location.
# Keep them near the top so it's easy to swap or parameterize later.

# thyroid = pd.read_csv('C:/Users/Lenovo/Desktop/navaneetha/Major Project/She-Health/backend/data/set_1/cleaned_dataset_Thyroid1.csv')
# cervical = pd.read_csv('C:/Users/Lenovo/Desktop/navaneetha/Major Project/She-Health/backend/data/set_1/cervical-cancer_csv.csv')
pcos = pd.read_csv('C:/Users/Lenovo/Desktop/navaneetha/Major Project/She-Health/backend/data/final_dataset/pcos.csv')
# endometriosis = pd.read_csv('C:/Users/Lenovo/Desktop/navaneetha/Major Project/She-Health/backend/data/set_1/structured_endometriosis_data.csv')

def _infer_problem_type(y) -> str:
    t = type_of_target(y)
    if t in ("continuous", "continuous-multioutput"):
        return "regression"
    return "classification"

def train_random_forest(df, target_col, output_path: str = 'predictions.csv', test_size: float = 0.3,
                        random_state: int = 42, grid_search: bool = False):
    """Train and evaluate a Random Forest classifier.

    Args:
        df: pandas DataFrame containing features and the target column.
        target_col: name of the target column in `df`.
        output_path: CSV path to write test predictions and probabilities.
        test_size: fraction of data to reserve for the test set.
        random_state: seed for reproducibility.
        grid_search: if True, run a small GridSearchCV (uses a tiny example grid).

    Returns:
        clf: fitted estimator
        out_df: DataFrame with test features + true/pred/(proba)
        metrics: dict with accuracy/precision/recall/f1/roc_auc
    """
    data = df.copy()
    data = data.dropna(subset=[target_col])
    X = data.drop(columns=[target_col])
    y = data[target_col]

    if _infer_problem_type(y) != "classification":
        raise ValueError("Provided target looks like a regression target. Expecting classification.")

    # Simple preprocessing steps:
    # 1) Convert object/categorical columns to integer codes (preserves ordinals as codes).
    # 2) Fill numeric missing values with column median.
    # 3) Expand remaining categorical columns via one-hot encoding.
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)

    # median imputation for numeric columns
    X = X.fillna(X.median(numeric_only=True))
    # one-hot encode any remaining categorical columns (safer for tree-based models too)
    X = pd.get_dummies(X, drop_first=True)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Choose estimator: simple RF or a small GridSearchCV wrapper
    if grid_search:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        clf = GridSearchCV(RandomForestClassifier(random_state=random_state), param_grid, cv=3, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute predicted probabilities when the estimator supports it.
    y_prob = None
    if hasattr(clf, "predict_proba"):
        try:
            y_prob = clf.predict_proba(X_test)
        except Exception:
            # If probability computation fails, continue without probabilities.
            y_prob = None
    # Metrics
    unique_labels = np.unique(y)
    binary = len(unique_labels) == 2
    average = "binary" if binary else "macro"

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

    roc = None
    if y_prob is not None:
        try:
            if binary:
                # take prob of positive class (assume class ordering puts positive at index 1)
                pos_prob = y_prob[:, 1]
                roc = roc_auc_score(y_test, pos_prob)
            else:
                lb = label_binarize(y_test, classes=unique_labels)
                roc = roc_auc_score(lb, y_prob, average="macro", multi_class="ovr")
        except Exception:
            roc = None

    # Prepare output DataFrame containing test features and prediction results.
    out_df = X_test.copy()
    out_df[f"{target_col}_true"] = y_test.values
    out_df[f"{target_col}_pred"] = y_pred
    if y_prob is not None:
        if binary:
            # positive-class probability (assumes sklearn ordering)
            out_df[f"{target_col}_proba_pos"] = y_prob[:, 1]
        else:
            # multi-class probabilities: one column per class
            for i, cls in enumerate(unique_labels):
                out_df[f"proba_{cls}"] = y_prob[:, i]

    # Persist predictions for later inspection.
    out_df.to_csv(output_path, index=False)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}" if roc is not None else "ROC-AUC: N/A")

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
    return clf, out_df, metrics


def train_generic_model(df, target_col, estimator, out_csv, out_pickle=None,
                        test_size: float = 0.3, random_state: int = 42):
    """Train a generic sklearn-like estimator with the same preprocessing and
    evaluation workflow used for RandomForest.

    Writes predictions CSV to `out_csv`. Returns the metrics dict.
    """
    data = df.copy()
    data = data.dropna(subset=[target_col])
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # same preprocessing as RF helper
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = pd.get_dummies(X, drop_first=True)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # fit estimator (accepts either an instantiated estimator or a class)
    est = estimator
    if callable(estimator) and not hasattr(estimator, "fit"):
        # constructor passed instead of instance
        est = estimator()

    # ensure reproducible seed when possible
    try:
        est.set_params(random_state=random_state)
    except Exception:
        pass

    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)

    # probabilities if available
    y_prob = None
    if hasattr(est, "predict_proba"):
        try:
            y_prob = est.predict_proba(X_test)
        except Exception:
            y_prob = None

    unique_labels = np.unique(y)
    binary = len(unique_labels) == 2
    average = "binary" if binary else "macro"

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

    roc = None
    if y_prob is not None:
        try:
            if binary:
                pos_prob = y_prob[:, 1]
                roc = roc_auc_score(y_test, pos_prob)
            else:
                lb = label_binarize(y_test, classes=unique_labels)
                roc = roc_auc_score(lb, y_prob, average="macro", multi_class="ovr")
        except Exception:
            roc = None

    # output dataframe
    out_df = X_test.copy()
    out_df[f"{target_col}_true"] = y_test.values
    out_df[f"{target_col}_pred"] = y_pred
    if y_prob is not None:
        if binary:
            out_df[f"{target_col}_proba_pos"] = y_prob[:, 1]
        else:
            for i, cls in enumerate(unique_labels):
                out_df[f"proba_{cls}"] = y_prob[:, i]

    try:
        out_df.to_csv(out_csv, index=False)
    except Exception:
        print(f"Warning: failed to write predictions to {out_csv}")
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}
    return metrics


if __name__ == "__main__":
    # Create a folder to store CSV outputs and pickled models
    OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)

    def pick_target(df, explicit: Optional[str] = None, candidates: Optional[list] = None,
                    contains: Optional[str] = None) -> str:
        """Choose a sensible target column from a dataframe.

        Selection order: explicit -> first matching candidate -> first column containing `contains`
        -> fallback to last column in the DataFrame.
        """
        if explicit and explicit in df.columns:
            return explicit
        if candidates:
            for c in candidates:
                if c in df.columns:
                    return c
        if contains:
            for c in df.columns:
                if contains in c.lower():
                    return c
        return df.columns[-1]

    # List of datasets to process: (short_name, dataframe, explicit_target, candidate_names, contains_substr)
    jobs = [
        # ("thyroid", thyroid, "binaryClass", None, None),
        # ("cervical", cervical, "Biopsy", None, None),
        ("pcos", pcos, "PCOS", None, None),
        # ("endometriosis", endometriosis, "Diagnosis", None, None),
    ]

    # Collect metrics across all datasets/models and iterate datasets to train
    metrics_records = []
    # Iterate datasets, train RF, save predictions and model.
    for name, df, explicit, candidates, contains in jobs:
        try:
            target = pick_target(df, explicit, candidates, contains)
            print(f"\n--- Training RF for {name}, target={target} ---")

            out_csv = os.path.join(OUT_DIR, f"{name}_rf_predictions.csv")
            clf, out_df, metrics = train_random_forest(df, target, output_path=out_csv)

            print(f"Saved predictions: {out_csv}")
            print("Metrics:", metrics)
            # Collect metrics to write once at the end
            perf_file = os.path.join(OUT_DIR, "rf_predictions_performance_measure.txt")
            # record RF metrics
            metrics_records.append({
                "dataset": name,
                "model": "random_forest",
                "target": target,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "metrics": metrics,
                "predictions": out_csv,
            })
            # Also train other classifiers with the same workflow
            models_to_run = []
            if XGBClassifier is not None:
                models_to_run.append(("xgboost", lambda: XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42)))
            if LGBMClassifier is not None:
                models_to_run.append(("lightgbm", lambda: LGBMClassifier(n_estimators=100, random_state=42)))
            if CatBoostClassifier is not None:
                models_to_run.append(("catboost", lambda: CatBoostClassifier(verbose=0, random_state=42)))
            # sklearn models (should be available)
            models_to_run.extend([
                ("svm", lambda: SVC(probability=True, random_state=42)),
                ("knn", lambda: KNeighborsClassifier()),
                ("nb", lambda: GaussianNB()),
            ])

            for mk, ctor in models_to_run:
                try:
                    mod_out_csv = os.path.join(OUT_DIR, f"{name}_{mk}_predictions.csv")
                    # Do not create pickle files per user request
                    mod_out_pickle = None
                    print(f"Training {mk} for dataset {name} (target={target})...")
                    m_metrics = train_generic_model(df, target, ctor, mod_out_csv, mod_out_pickle)
                    print(f"{mk} metrics:", m_metrics)
                    # collect metrics for this model run (no pickle saved)
                    metrics_records.append({
                        "dataset": name,
                        "model": mk,
                        "target": target,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "metrics": m_metrics,
                        "predictions": mod_out_csv,
                    })
                except Exception as e:
                    print(f"Failed to train {mk} on {name}: {e}")
        except Exception as e:
            print(f"Failed to train RF for {name}: {e}")
    # After all datasets and models finished, write the performance file once
    try:
        perf_file = os.path.join(OUT_DIR, "rf_predictions_performance_measure.txt")
        with open(perf_file, "w", encoding="utf-8") as pf:
            for rec in metrics_records:
                pf.write(f"=== {rec['dataset']} - {rec['model']} ===\n")
                pf.write(f"Target: {rec['target']}\n")
                pf.write(f"Timestamp (UTC): {rec['timestamp']}\n")
                pf.write(f"Predictions CSV: {rec.get('predictions','')}\n")
                for mk, mv in rec['metrics'].items():
                    pf.write(f"{mk}: {mv}\n")
                pf.write("\n")
        print(f"Wrote performance summary to {perf_file}")
    except Exception as e:
        print(f"Warning: failed to write summary performance file: {e}")


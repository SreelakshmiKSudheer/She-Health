from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.base import is_classifier, clone
from sklearn.utils.multiclass import type_of_target
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

"""
featureImportance.py

Utilities to train a Random Forest and compute feature importances.
"""

import matplotlib.pyplot as plt
import os

# directory to save outputs (CSVs/plots)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
thyroid = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\cleaned_dataset_Thyroid1.csv')
cervical = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\cervical-cancer_csv.csv')
pcos = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\PCOS_data.csv')
endometriosis = pd.read_csv('C:\\Users\\user\\SreelakshmiK\\personal\\Projects\\She-Health\\backend\\data\\structured_endometriosis_data.csv')


def _infer_problem_type(y) -> str:
    t = type_of_target(y)
    if t in ("continuous", "continuous-multioutput"):
        return "regression"
    return "classification"


def train_random_forest(
    X,
    y,
    feature_names: Optional[list] = None,
    problem_type: Optional[str] = None,
    cv: int = 5,
    param_grid: Optional[Dict[str, list]] = None,
    n_jobs: int = -1,
    random_state: int = 42,
    n_estimators: int = 100,
    scoring: Optional[str] = None,
    compute_permutation: bool = True,
    perm_n_repeats: int = 10,
) -> Dict[str, Any]:
    """
    Train a Random Forest (classifier or regressor) and return model and importances.

    Returns a dict:
      - model: fitted estimator
      - importances: pandas.DataFrame with columns ['feature','importance'] sorted desc
      - permutation_importances: pandas.DataFrame (if compute_permutation True)
      - cv_results: GridSearchCV.cv_results_ (if param_grid provided)
    """
    # Prepare X, feature names
    if hasattr(X, "values"):
        X_values = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_values = np.asarray(X)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X_values.shape[1])]

    # Infer problem type if not provided
    if problem_type is None:
        problem_type = _infer_problem_type(y)

    # Choose estimator
    if problem_type == "regression":
        estimator = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators, n_jobs=n_jobs)
        if scoring is None:
            scoring = "neg_mean_squared_error"
    else:
        estimator = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, n_jobs=n_jobs)
        if scoring is None:
            # use ROC AUC for binary, accuracy otherwise
            try:
                t = type_of_target(y)
                scoring = "roc_auc" if t == "binary" else "accuracy"
            except Exception:
                scoring = "accuracy"

    result: Dict[str, Any] = {}

    # Optional hyperparameter tuning
    if param_grid:
        gs = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=True)
        gs.fit(X_values, y)
        model = gs.best_estimator_
        result["cv_results"] = gs.cv_results_
        result["best_params"] = gs.best_params_
    else:
        model = clone(estimator)
        model.set_params(random_state=random_state)
        model.fit(X_values, y)

    result["model"] = model

    # Built-in feature importances
    try:
        importances = model.feature_importances_
    except Exception:
        importances = np.zeros(X_values.shape[1])

    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    result["importances"] = df_imp

    # Permutation importances (more reliable)
    if compute_permutation:
        try:
            perm = permutation_importance(model, X_values, y, n_repeats=perm_n_repeats, random_state=random_state, n_jobs=n_jobs, scoring=scoring)
            perm_means = perm.importances_mean
            df_perm = pd.DataFrame(
                {"feature": feature_names, "importance": perm_means}
            ).sort_values("importance", ascending=False).reset_index(drop=True)
            result["permutation_importances"] = df_perm
            result["permutation_result_obj"] = perm
        except Exception:
            # if permutation fails (e.g., scoring incompatible), skip
            result["permutation_importances"] = None
            result["permutation_result_obj"] = None

    return result


if __name__ == "__main__":


    def process_dataset(df, name, target_col: Optional[str] = None, target_candidates: Optional[list] = None,
                        contains: Optional[str] = None, n_estimators: int = 200):
        """
        Generic processing + RF training for a dataset.
        - df: DataFrame
        - name: short name used for prints and output filenames
        - target_col: explicit target column name (preferred)
        - target_candidates: list of candidate names to try in order
        - contains: substring to search for in column names (case-insensitive)
        """
        try:
            print(f"\nTraining Random Forest on {name} dataset...")
            dfc = df.copy()
            dfc.columns = [c.strip() for c in dfc.columns]

            # Determine target column
            if target_col is None:
                if target_candidates:
                    for c in target_candidates:
                        if c in dfc.columns:
                            target_col = c
                            break
                if target_col is None and contains:
                    for c in dfc.columns:
                        if contains in c.lower():
                            target_col = c
                            break
                if target_col is None:
                    target_col = dfc.columns[-1]  # fallback

            print(f"Using target column: {target_col}")

            # Convert columns to numeric where possible
            for col in dfc.columns:
                dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

            dfc = dfc.dropna(subset=[target_col])

            X = dfc.drop(columns=[target_col])
            y = dfc[target_col]

            # Simple imputation for features
            X = X.fillna(X.median())

            try:
                y = y.astype(int)
            except Exception:
                pass

            rf_out = train_random_forest(X, y, compute_permutation=True, n_estimators=n_estimators)

            print(f"\n{name.capitalize()} Random Forest feature precedence (built-in importance):")
            print(rf_out['importances'].to_string(index=False))

            if rf_out.get('permutation_importances') is not None:
                print(f"\n{name.capitalize()} Random Forest permutation importances:")
                print(rf_out['permutation_importances'].to_string(index=False))

            # save importances
            try:
                rf_out['importances'].to_csv(os.path.join(OUTPUT_DIR, f'{name}_rf_importances.csv'), index=False)
                if rf_out.get('permutation_importances') is not None:
                    rf_out['permutation_importances'].to_csv(os.path.join(OUTPUT_DIR, f'{name}_rf_permutation_importances.csv'), index=False)
            except Exception:
                print(f'Warning: Failed to save {name} RF importances to CSV')
        except Exception as e:
            print(f"Failed to train Random Forest on {name} dataset: {e}")


    if __name__ == "__main__":
        # thyroid: explicit binaryClass target
        process_dataset(thyroid, "thyroid", target_col="binaryClass", n_estimators=200)

        # cervical: prefer Biopsy, then Dx, else last column
        process_dataset(cervical, "cervical", target_candidates=["Biopsy", "Dx"], n_estimators=200)

        # PCOS: search for column names containing 'pcos' (case-insensitive) or common variants
        process_dataset(pcos, "pcos", target_candidates=["PCOS (Y/N)", "PCOS"], contains="pcos", n_estimators=200)

        # endometriosis: prefer 'Diagnosis' else last column
        process_dataset(endometriosis, "endometriosis", target_candidates=["Diagnosis"], n_estimators=200)

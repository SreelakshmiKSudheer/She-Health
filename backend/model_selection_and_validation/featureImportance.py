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


def plot_importances(importances_df: pd.DataFrame, top_n: int = 20, figsize: Tuple[int, int] = (8, 6), title: Optional[str] = None):
    """
    Plot horizontal bar chart of top_n importances DataFrame (columns: feature, importance).
    """
    df = importances_df.copy().head(top_n)
    plt.figure(figsize=figsize)
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.xlabel("Importance")
    plt.tight_layout()
    if title:
        plt.title(title)
    plt.show()


if __name__ == "__main__":

    # Train a Decision Tree on the thyroid dataset and print feature precedence
    def train_decision_tree_on_thyroid(df: pd.DataFrame, target_col: str = "binaryClass", random_state: int = 42, max_depth: Optional[int] = None):
        df = df.copy()
        # normalize column names
        df.columns = [c.strip() for c in df.columns]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")

        # Convert all columns except target to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # simple imputation: fill numeric NaNs with median
        X = X.fillna(X.median())

        # ensure integer labels if binary
        try:
            y = y.astype(int)
        except Exception:
            pass

        # train/test split for a quick check (not strictly necessary for importances)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if len(set(y))>1 else None)

        clf = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
        clf.fit(X_train, y_train)

        importances = clf.feature_importances_
        df_imp = pd.DataFrame({"feature": X.columns, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)

        # Print features in precedence order
        print("\nDecision Tree feature precedence (highest -> lowest):")
        for idx, row in df_imp.iterrows():
            print(f"{idx+1}. {row['feature']}: {row['importance']:.6f}")

        return {"model": clf, "importances": df_imp}

    try:
        print("\nTraining Decision Tree on thyroid dataset...")
        dt_out = train_decision_tree_on_thyroid(thyroid)
        # also show top 10
        print('\nTop 10 features:')
        print(dt_out['importances'].head(10).to_string(index=False))
    except Exception as e:
        print(f"Failed to train Decision Tree on thyroid dataset: {e}")

    # Train a Random Forest on the thyroid dataset and print feature precedence
    try:
        print("\nTraining Random Forest on thyroid dataset...")
        df = thyroid.copy()
        df.columns = [c.strip() for c in df.columns]
        target_col = "binaryClass"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[target_col])

        X_rf = df.drop(columns=[target_col])
        y_rf = df[target_col]

        X_rf = X_rf.fillna(X_rf.median())
        try:
            y_rf = y_rf.astype(int)
        except Exception:
            pass

        rf_out = train_random_forest(X_rf, y_rf, compute_permutation=True, n_estimators=200)

        print("\nRandom Forest feature precedence (built-in importance):")
        print(rf_out['importances'].to_string(index=False))

        if rf_out.get('permutation_importances') is not None:
            print("\nRandom Forest permutation importances:")
            print(rf_out['permutation_importances'].to_string(index=False))
    except Exception as e:
        print(f"Failed to train Random Forest on thyroid dataset: {e}")

    # Train a Random Forest on the cervical dataset and print feature precedence
    try:
        print("\nTraining Random Forest on cervical dataset...")
        dfc = cervical.copy()
        dfc.columns = [c.strip() for c in dfc.columns]
        # choose target column (use 'Biopsy' if present, otherwise try common Dx columns)
        if 'Biopsy' in dfc.columns:
            target_col = 'Biopsy'
        elif 'Dx' in dfc.columns:
            target_col = 'Dx'
        else:
            # fallback to last column
            target_col = dfc.columns[-1]

        print(f"Using target column: {target_col}")

        for col in dfc.columns:
            dfc[col] = pd.to_numeric(dfc[col], errors='coerce')

        dfc = dfc.dropna(subset=[target_col])

        X_c = dfc.drop(columns=[target_col])
        y_c = dfc[target_col]

        X_c = X_c.fillna(X_c.median())
        try:
            y_c = y_c.astype(int)
        except Exception:
            pass

        rf_out_c = train_random_forest(X_c, y_c, compute_permutation=True, n_estimators=200)

        print("\nCervical Random Forest feature precedence (built-in importance):")
        print(rf_out_c['importances'].to_string(index=False))

        if rf_out_c.get('permutation_importances') is not None:
            print("\nCervical Random Forest permutation importances:")
            print(rf_out_c['permutation_importances'].to_string(index=False))
    except Exception as e:
        print(f"Failed to train Random Forest on cervical dataset: {e}")

    # Train a Random Forest on the PCOS dataset and print feature precedence
    try:
        print("\nTraining Random Forest on PCOS dataset...")
        dfp = pcos.copy()
        dfp.columns = [c.strip() for c in dfp.columns]

        # detect a sensible target column: look for 'pcos' in column name, else try common variants
        target_col = None
        for c in dfp.columns:
            if 'pcos' in c.lower():
                target_col = c
                break
        if target_col is None:
            # try exact common name
            for candidate in ['PCOS (Y/N)', 'PCOS', 'PCOS (Y/N)']:
                if candidate in dfp.columns:
                    target_col = candidate
                    break
        if target_col is None:
            # fallback to a column named like 'PCOS (Y/N)' with variable whitespace
            for c in dfp.columns:
                if 'pcos' in c.replace(' ', '').lower():
                    target_col = c
                    break
        if target_col is None:
            raise ValueError('Could not find a PCOS target column in PCOS dataset')

        print(f"Using target column: {target_col}")

        for col in dfp.columns:
            dfp[col] = pd.to_numeric(dfp[col], errors='coerce')

        dfp = dfp.dropna(subset=[target_col])

        X_p = dfp.drop(columns=[target_col])
        y_p = dfp[target_col]

        X_p = X_p.fillna(X_p.median())
        try:
            y_p = y_p.astype(int)
        except Exception:
            pass

        rf_out_p = train_random_forest(X_p, y_p, compute_permutation=True, n_estimators=200)

        print("\nPCOS Random Forest feature precedence (built-in importance):")
        print(rf_out_p['importances'].to_string(index=False))

        if rf_out_p.get('permutation_importances') is not None:
            print("\nPCOS Random Forest permutation importances:")
            print(rf_out_p['permutation_importances'].to_string(index=False))
    except Exception as e:
        print(f"Failed to train Random Forest on PCOS dataset: {e}")

    # Train a Random Forest on the endometriosis dataset and print feature precedence
    try:
        print("\nTraining Random Forest on endometriosis dataset...")
        dfe = endometriosis.copy()
        dfe.columns = [c.strip() for c in dfe.columns]

        # prefer a Diagnosis-like column
        if 'Diagnosis' in dfe.columns:
            target_col = 'Diagnosis'
        else:
            target_col = dfe.columns[-1]

        print(f"Using target column: {target_col}")

        for col in dfe.columns:
            dfe[col] = pd.to_numeric(dfe[col], errors='coerce')

        dfe = dfe.dropna(subset=[target_col])

        X_e = dfe.drop(columns=[target_col])
        y_e = dfe[target_col]

        X_e = X_e.fillna(X_e.median())
        try:
            y_e = y_e.astype(int)
        except Exception:
            pass

        rf_out_e = train_random_forest(X_e, y_e, compute_permutation=True, n_estimators=200)

        print("\nEndometriosis Random Forest feature precedence (built-in importance):")
        print(rf_out_e['importances'].to_string(index=False))

        if rf_out_e.get('permutation_importances') is not None:
            print("\nEndometriosis Random Forest permutation importances:")
            print(rf_out_e['permutation_importances'].to_string(index=False))
    except Exception as e:
        print(f"Failed to train Random Forest on endometriosis dataset: {e}")
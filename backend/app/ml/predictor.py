from pathlib import Path
import math
import re

import joblib
import pandas as pd


class SheHealthPredictor:
    # Contract-level disease registry based on the currently trained and provided
    # backend models/artifacts.
    DISEASE_MODELS = {
        "PCOS": ["pcos_model.pkl"],
        "Endometriosis": ["endometriosis_model.pkl"],
        "Cervical": ["cervical_cancer_model.pkl", "cervical_canvcer_model.pkl"],
    }

    # Runtime thresholds explicitly selected for each disease model.
    DISEASE_THRESHOLDS = {
        "PCOS": 0.25,
        "Endometriosis": 0.35,
        "Cervical": 0.10,
    }

    def __init__(self):
        self.model_dir = Path(__file__).resolve().parent / "models"
        self.models = {}
        self.unavailable_diseases = set()
        self.load_errors = {}

        for disease, candidates in self.DISEASE_MODELS.items():
            model = self._load_first_available_model(candidates)
            if isinstance(model, dict) and "__load_error__" in model:
                self.unavailable_diseases.add(disease)
                self.load_errors[disease] = model["__load_error__"]
            elif model is None:
                self.unavailable_diseases.add(disease)
            else:
                self.models[disease] = self._normalize_artifact(model)

    def _normalize_artifact(self, artifact):
        # Supports both raw estimators and wrapped artifacts dumped as:
        # {"model": estimator, "features": [...], "threshold": 0.25}
        if isinstance(artifact, dict) and "model" in artifact:
            return {
                "estimator": artifact["model"],
                "features": artifact.get("features") or [],
                "threshold": artifact.get("threshold"),
            }
        return {
            "estimator": artifact,
            "features": [],
            "threshold": None,
        }

    def _resolve_threshold(self, disease: str, artifact_threshold):
        if disease in self.DISEASE_THRESHOLDS:
            return float(self.DISEASE_THRESHOLDS[disease])
        if artifact_threshold is not None:
            return float(artifact_threshold)
        return 0.5

    def _load_first_available_model(self, filenames):
        last_error = None
        for name in filenames:
            path = self.model_dir / name
            if path.exists():
                try:
                    return joblib.load(path)
                except Exception as exc:
                    last_error = f"{name}: {exc.__class__.__name__}"
                    continue
        if last_error is not None:
            return {"__load_error__": last_error}
        return None

    def predict_all_diseases(self, feature_dict: dict):
        results = {}

        for disease in self.DISEASE_MODELS:
            artifact = self.models.get(disease)
            if artifact is None:
                load_error = self.load_errors.get(disease)
                results[disease] = {
                    "probability": 0.0,
                    "risk_level": 0,
                    "label": "Unavailable",
                    "category_name": "Unavailable",
                    "status": "model_unavailable",
                    "details": load_error,
                }
                continue

            try:
                estimator = artifact["estimator"]
                threshold = self._resolve_threshold(disease, artifact.get("threshold"))
                model_features = artifact.get("features") or []

                model_df, coverage = self._build_model_dataframe(feature_dict, model_features)

                probability_pct = self._predict_probability_pct(estimator, model_df)
                risk_level = self._predict_risk_level(estimator, model_df, probability_pct)
                label = self._risk_label(risk_level)

                results[disease] = {
                    "probability": round(probability_pct, 2),
                    "risk_level": int(risk_level),
                    "label": label,
                    "category_name": label,
                    "status": "ok",
                    "threshold": threshold,
                    "predicted_positive": bool(
                        (probability_pct / 100.0) >= threshold
                    ),
                    "feature_coverage": coverage,
                }
            except Exception as exc:
                # Keep the contract stable even when one model fails at runtime.
                results[disease] = {
                    "probability": 0.0,
                    "risk_level": 0,
                    "label": "Unavailable",
                    "category_name": "Unavailable",
                    "status": f"prediction_error: {exc.__class__.__name__}",
                }

        return results

    def get_feature_coverage_report(self, feature_dict: dict):
        report = {}
        for disease in self.DISEASE_MODELS:
            artifact = self.models.get(disease)
            if artifact is None:
                report[disease] = {
                    "status": "model_unavailable",
                    "details": self.load_errors.get(disease),
                }
                continue

            _, coverage = self._build_model_dataframe(
                feature_dict, artifact.get("features") or []
            )
            report[disease] = {
                "status": "ok",
                "threshold": self._resolve_threshold(disease, artifact.get("threshold")),
                "feature_coverage": coverage,
            }
        return report

    def _build_model_dataframe(self, feature_dict: dict, model_features: list):
        # If artifact does not provide an expected feature list, pass all mapped features.
        if not model_features:
            frame = pd.DataFrame([feature_dict]).fillna(0)
            coverage = {
                "provided": len(feature_dict),
                "expected": len(feature_dict),
                "missing": 0,
                "missing_features": [],
            }
            return frame, coverage

        normalized_map = {}
        for key, value in feature_dict.items():
            for alias in self._feature_aliases(key):
                normalized_map[alias] = value

        row = {}
        provided = 0
        missing_features = []
        for feature_name in model_features:
            # Prefer exact key first, then normalized matching to tolerate minor spacing/case drift.
            if feature_name in feature_dict:
                row[feature_name] = feature_dict[feature_name]
                provided += 1
                continue

            matched, value = self._find_feature_value(feature_name, normalized_map)
            if matched:
                row[feature_name] = value
                provided += 1
            else:
                row[feature_name] = 0
                missing_features.append(feature_name)

        frame = pd.DataFrame([row], columns=model_features).fillna(0)
        coverage = {
            "provided": provided,
            "expected": len(model_features),
            "missing": len(model_features) - provided,
            "missing_features": missing_features,
        }
        return frame, coverage

    def _normalize_feature_name(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", str(value).strip().lower())
        return " ".join(cleaned.split())

    def _feature_aliases(self, key: str):
        aliases = set()
        normalized = self._normalize_feature_name(key)
        aliases.add(normalized)

        # Strip known disease suffixes in questionnaire feature names, e.g.:
        # Age_pcos -> Age, irregular_missed_periods_endo -> irregular_missed_periods
        suffix_stripped = re.sub(
            r"_(pcos|endo|endometriosis|cervical_cancer)$",
            "",
            str(key),
            flags=re.IGNORECASE,
        )
        aliases.add(self._normalize_feature_name(suffix_stripped))

        # Some questionnaires store names in snake_case; model columns use spaces.
        aliases.add(self._normalize_feature_name(str(key).replace("_", " ")))

        return aliases

    def _find_feature_value(self, model_feature: str, normalized_map: dict):
        target = self._normalize_feature_name(model_feature)

        # 1) Exact alias match
        if target in normalized_map:
            return True, normalized_map[target]

        # 2) Disease-derived boolean features for cervical model
        derived = self._derive_boolean_feature(target, normalized_map)
        if derived is not None:
            return True, derived

        # 3) Fuzzy token overlap match (handles e.g., "irregular missed periods"
        # vs "irregular or missed periods")
        target_tokens = self._feature_tokens(target)
        best_key = None
        best_score = 0
        for candidate in normalized_map.keys():
            candidate_tokens = self._feature_tokens(candidate)
            if not candidate_tokens:
                continue
            score = len(target_tokens & candidate_tokens)
            if score > best_score:
                best_score = score
                best_key = candidate

        if best_key is not None and best_score >= 2:
            return True, normalized_map[best_key]

        return False, 0.0

    def _feature_tokens(self, text: str):
        # Ignore connector words that do not carry semantic meaning for matching.
        stop = {"and", "or", "of", "the", "during", "number", "time", "since"}
        tokens = set(self._normalize_feature_name(text).split())
        return {t for t in tokens if t not in stop}

    def _derive_boolean_feature(self, target: str, normalized_map: dict):
        def val(key):
            try:
                return float(normalized_map.get(key, 0.0))
            except Exception:
                return 0.0

        if target == "smokes":
            if target in normalized_map:
                return val(target)
            return 1.0 if (val("smokes years") > 0 or val("smokes packs year") > 0) else 0.0

        if target == "iud":
            if target in normalized_map:
                return val(target)
            return 1.0 if val("iud years") > 0 else 0.0

        if target == "stds":
            if target in normalized_map:
                return val(target)
            if val("stds number") > 0 or val("stds number of diagnosis") > 0:
                return 1.0

            std_subfeatures = [
                "stds condylomatosis",
                "stds cervical condylomatosis",
                "stds vaginal condylomatosis",
                "stds vulvo perineal condylomatosis",
                "stds syphilis",
                "stds pelvic inflammatory disease",
                "stds genital herpes",
                "stds molluscum contagiosum",
                "stds aids",
                "stds hiv",
                "stds hepatitis b",
                "stds hpv",
            ]
            return 1.0 if any(val(k) > 0 for k in std_subfeatures) else 0.0

        # Numeric cervical fields fallback when only boolean indicators exist.
        if target in {"smokes years", "smokes packs year"}:
            smokes_bool = self._derive_boolean_feature("smokes", normalized_map)
            if smokes_bool is not None:
                return float(smokes_bool)

        if target == "iud years":
            iud_bool = self._derive_boolean_feature("iud", normalized_map)
            if iud_bool is not None:
                return float(iud_bool)

        if target in {"stds number", "stds number of diagnosis"}:
            stds_bool = self._derive_boolean_feature("stds", normalized_map)
            if stds_bool is not None:
                return float(stds_bool)

        return None

    def _predict_probability_pct(self, model, df: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            if proba is not None:
                # Binary classifiers usually return Nx2. For edge cases, fallback to last column.
                if len(proba[0]) >= 2:
                    return float(proba[0][1] * 100)
                return float(proba[0][-1] * 100)

        if hasattr(model, "decision_function"):
            score = float(model.decision_function(df)[0])
            # Convert decision score to a pseudo-probability via sigmoid.
            prob = 1.0 / (1.0 + math.exp(-score))
            return prob * 100

        if hasattr(model, "predict"):
            pred = float(model.predict(df)[0])
            return 100.0 if pred >= 1 else 0.0

        return 0.0

    def _predict_risk_level(self, model, df: pd.DataFrame, probability_pct: float) -> int:
        # Preferred path: custom category method packaged inside the model.
        if hasattr(model, "predict_risk_category"):
            raw_level = model.predict_risk_category(df)[0]
            return self._normalize_level(raw_level, probability_pct)

        # Fallback path: map from probability thresholds to 1-5 buckets.
        return self._probability_to_level(probability_pct)

    def _normalize_level(self, raw_level, probability_pct: float) -> int:
        if isinstance(raw_level, str):
            label_map = {
                "very low": 1,
                "low": 2,
                "moderate": 3,
                "high": 4,
                "very high": 5,
            }
            mapped = label_map.get(raw_level.strip().lower())
            if mapped is not None:
                return mapped
            return self._probability_to_level(probability_pct)

        try:
            level = int(raw_level)
            if 1 <= level <= 5:
                return level
        except Exception:
            pass
        return self._probability_to_level(probability_pct)

    def _probability_to_level(self, probability_pct: float) -> int:
        if probability_pct <= 20:
            return 1
        if probability_pct <= 40:
            return 2
        if probability_pct <= 60:
            return 3
        if probability_pct <= 80:
            return 4
        return 5

    def _risk_label(self, level: int) -> str:
        labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
        return labels.get(level, "Unknown")
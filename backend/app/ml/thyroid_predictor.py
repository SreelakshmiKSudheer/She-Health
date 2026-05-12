from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


class ThyroidPredictor:
    MODEL_FILE = "thyroid_model.pkl"
    FEATURE_FALLBACKS = [
        "age",
        "sex",
        "on thyroxine",
        "query on thyroxine",
        "on antithyroid medication",
        "sick",
        "pregnant",
        "thyroid surgery",
        "I131 treatment",
        "query hypothyroid",
        "query hyperthyroid",
        "lithium",
        "goitre",
        "tumor",
        "hypopituitary",
        "psych",
        "TSH measured",
        "TSH",
        "T3 measured",
        "T3",
        "TT4 measured",
        "TT4",
        "T4U measured",
        "T4U",
        "FTI measured",
        "FTI",
    ]

    def __init__(self):
        self.model_dir = Path(__file__).resolve().parent / "models"
        self.model = None
        self.load_error: str | None = None
        self._load_model()

    def _load_model(self):
        model_path = self.model_dir / self.MODEL_FILE
        if not model_path.exists():
            self.load_error = f"{self.MODEL_FILE} not found"
            return

        try:
            self.model = joblib.load(model_path)
        except Exception as exc:
            self.load_error = f"{exc.__class__.__name__}: {exc}"
            self.model = None

    @staticmethod
    def _normalize_feature_name(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    def _get_expected_features(self) -> list[str]:
        if isinstance(self.model, dict):
            features = self.model.get("features") or []
            if features:
                return list(features)

        for attribute_name in ("feature_names_in_", "feature_names_in"):
            features = getattr(self.model, attribute_name, None)
            if features is not None:
                return [str(feature) for feature in features]

        return list(self.FEATURE_FALLBACKS)

    def _build_frame(self, feature_dict: Dict[str, Any]) -> tuple[pd.DataFrame, dict]:
        expected_features = self._get_expected_features()
        normalized = {self._normalize_feature_name(key): value for key, value in feature_dict.items()}

        row = {}
        provided = 0
        for feature_name in expected_features:
            normalized_name = self._normalize_feature_name(feature_name)
            if normalized_name in normalized:
                row[feature_name] = normalized[normalized_name]
                provided += 1
            else:
                row[feature_name] = 0.0

        frame = pd.DataFrame([row], columns=expected_features).fillna(0)
        coverage = {
            "provided": provided,
            "expected": len(expected_features),
            "missing": len(expected_features) - provided,
            "missing_features": [feature for feature in expected_features if self._normalize_feature_name(feature) not in normalized],
        }
        return frame, coverage

    @staticmethod
    def _risk_level(probability_pct: float) -> int:
        if probability_pct >= 70:
            return 5
        if probability_pct >= 40:
            return 4
        if probability_pct >= 20:
            return 3
        if probability_pct >= 10:
            return 2
        return 1

    @staticmethod
    def _risk_label(risk_level: int) -> str:
        return {
            5: "Very High Risk",
            4: "High Risk",
            3: "Moderate Risk",
            2: "Low Risk",
            1: "Very Low Risk",
        }.get(risk_level, "Unknown")

    def _fallback_probability(self, feature_dict: Dict[str, Any]) -> float:
        weighted_flags = [
            "on thyroxine",
            "query on thyroxine",
            "on antithyroid medication",
            "thyroid surgery",
            "I131 treatment",
            "query hypothyroid",
            "query hyperthyroid",
            "lithium",
            "goitre",
            "tumor",
            "hypopituitary",
            "psych",
            "sick",
            "pregnant",
        ]

        score = 0.0
        for feature_name in weighted_flags:
            value = feature_dict.get(feature_name, 0)
            try:
                score += 1.5 if float(value) > 0 else 0.0
            except Exception:
                score += 1.5 if str(value).strip().lower() in {"1", "true", "yes", "y"} else 0.0

        for feature_name in ("TSH", "T3", "TT4", "T4U", "FTI"):
            value = feature_dict.get(feature_name, 0)
            try:
                numeric = abs(float(value))
            except Exception:
                numeric = 0.0
            if numeric:
                score += min(1.5, numeric / 10.0)

        max_score = (len(weighted_flags) * 1.5) + (5 * 1.5)
        return min(100.0, (score / max_score) * 100.0 if max_score else 0.0)

    def predict(self, feature_dict: Dict[str, Any]) -> dict:
        frame, coverage = self._build_frame(feature_dict)

        if self.model is not None:
            try:
                estimator = self.model.get("model") if isinstance(self.model, dict) else self.model
                threshold = None
                if isinstance(self.model, dict):
                    threshold = self.model.get("threshold")

                if hasattr(estimator, "predict_proba"):
                    probability_pct = float(estimator.predict_proba(frame)[0][1]) * 100.0
                elif hasattr(estimator, "decision_function"):
                    score = float(estimator.decision_function(frame)[0])
                    probability_pct = float(100.0 / (1.0 + pow(2.718281828, -score)))
                else:
                    probability_pct = self._fallback_probability(feature_dict)

                risk_level = self._risk_level(probability_pct)
                label = self._risk_label(risk_level)
                predicted_positive = threshold is not None and (probability_pct / 100.0) >= float(threshold)

                return {
                    "Thyroid": {
                        "probability": round(probability_pct, 2),
                        "risk_level": risk_level,
                        "category_name": label,
                        "label": label,
                        "status": "ok",
                        "threshold": threshold,
                        "predicted_positive": predicted_positive,
                        "feature_coverage": coverage,
                    }
                }
            except Exception as exc:
                return {
                    "Thyroid": {
                        "probability": 0.0,
                        "risk_level": 1,
                        "category_name": "Very Low Risk",
                        "label": "Very Low Risk",
                        "status": f"prediction_error: {exc.__class__.__name__}",
                        "details": str(exc),
                        "feature_coverage": coverage,
                    }
                }

        probability_pct = self._fallback_probability(feature_dict)
        risk_level = self._risk_level(probability_pct)
        label = self._risk_label(risk_level)
        return {
            "Thyroid": {
                "probability": round(probability_pct, 2),
                "risk_level": risk_level,
                "category_name": label,
                "label": label,
                "status": "fallback_model_unavailable",
                "details": self.load_error,
                "feature_coverage": coverage,
            }
        }

    def get_feature_coverage_report(self, feature_dict: Dict[str, Any]):
        _, coverage = self._build_frame(feature_dict)
        return {"Thyroid": {"status": "ok" if self.model is not None else "fallback_model_unavailable", "details": self.load_error, "feature_coverage": coverage}}

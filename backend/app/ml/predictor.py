import joblib
import pandas as pd
import os

class SheHealthPredictor:
    def __init__(self):
        model_path = "app/ml/models/"
        # Loading your 4 specific models
        self.models = {
            "PCOS": joblib.load(os.path.join(model_path, "pcos_model.pkl")),
            "Endometriosis": joblib.load(os.path.join(model_path, "endometriosis_model.pkl")),
            "Cervical": joblib.load(os.path.join(model_path, "cervical_canvcer_model.pkl"))
        }

    def predict_all_diseases(self, feature_dict: dict):
        # 1. Convert dictionary to the DataFrame format your models expect
        df = pd.DataFrame([feature_dict]).fillna(0)
        
        results = {}
        for name, model in self.models.items():
            # 2. Extract probability and level directly from your model's output
            # I am assuming your model has a custom method or you're using 
            # the standard predict_proba + your custom logic bundled in the pkl
            prob = model.predict_proba(df)[0][1] * 100
            
            # Since you said the predictor itself outputs the risk category:
            risk_lvl = model.predict_risk_category(df)[0] # Custom method name example
            
            results[name] = {
                "probability": round(prob, 2),
                "risk_level": int(risk_lvl),
                "category_name": self._get_label(risk_lvl)
            }
        return results

    def _get_label(self, level):
        # This mapping is just for the UI display string
        labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
        return labels.get(level, "Unknown")
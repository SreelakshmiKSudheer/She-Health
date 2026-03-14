import json
import os
import re
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


def read_csv_robust(path: Path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def norm(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", " ", str(s).strip().lower())
    return " ".join(s.split())


def token_set(s: str):
    return set(norm(s).split())


def best_column_from_feature(feature_name: str, columns):
    # Strip disease suffixes used in questionnaire mappings
    base = re.sub(r"_(pcos|endo|endometriosis|cervical_cancer)$", "", feature_name, flags=re.IGNORECASE)
    base_norm = norm(base.replace("_", " "))

    cols_norm = {c: norm(c) for c in columns}
    exact = [c for c, n in cols_norm.items() if n == base_norm]
    if exact:
        return exact[0]

    # Fuzzy by token overlap
    target_tokens = token_set(base_norm)
    best_col = None
    best_score = 0
    for c, n in cols_norm.items():
        t = token_set(n)
        if not t or not target_tokens:
            continue
        score = len(target_tokens & t)
        if score > best_score:
            best_score = score
            best_col = c
    if best_score >= 2:
        return best_col
    return None


def safe_float(v):
    try:
        if pd.isna(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def get_probability(model, x_df):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x_df)
        if len(p[0]) >= 2:
            return float(p[0][1])
        return float(p[0][-1])
    if hasattr(model, "predict"):
        return float(model.predict(x_df)[0])
    return 0.0


def build_model_input(row, model_features):
    row_norm = {norm(k): row[k] for k in row.index}
    out = {}
    for f in model_features:
        key = norm(f)
        out[f] = safe_float(row_norm.get(key, 0.0))
    return out


def map_questionnaire_responses(row, questions, disease_key):
    # disease_key in {"pcos", "endo", "cervical_cancer"}
    responses = []
    cols = list(row.index)

    for q in questions:
        q_type = q.get("q_type")
        q_id = q.get("id")

        # input questions: derive answer from direct_mappings by disease suffix
        if q_type == "input":
            direct = q.get("direct_mappings") or []
            mapped_items = []
            for m in direct:
                fname = m.get("feature_name", "")
                if not fname.lower().endswith(disease_key):
                    continue
                col = best_column_from_feature(fname, cols)
                if not col:
                    continue
                mapped_items.append({
                    "feature_name": fname,
                    "source_column": col,
                    "value": row[col],
                })
            if mapped_items:
                responses.append({
                    "question_id": q_id,
                    "q_type": q_type,
                    "input_values": mapped_items,
                })
            continue

        options = q.get("options") or []
        if not options:
            continue

        option_scores = []
        for opt in options:
            score = 0
            mapped = []
            for m in opt.get("mappings", []):
                fname = m.get("feature_name", "")
                if not fname.lower().endswith(disease_key):
                    continue
                col = best_column_from_feature(fname, cols)
                if not col:
                    continue
                expected = safe_float(m.get("feature_value", 0.0))
                actual = safe_float(row[col])
                if abs(actual - expected) < 1e-9:
                    score += 1
                    mapped.append({
                        "feature_name": fname,
                        "source_column": col,
                        "expected": expected,
                        "actual": actual,
                    })
            option_scores.append((score, opt.get("id"), mapped, opt.get("text")))

        if q_type == "single_select":
            option_scores.sort(key=lambda x: x[0], reverse=True)
            top = option_scores[0] if option_scores else None
            if top and top[0] > 0:
                responses.append({
                    "question_id": q_id,
                    "q_type": q_type,
                    "selected_option_ids": [top[1]],
                    "selected_option_texts": [top[3]],
                    "matched_mappings": top[2],
                })
        elif q_type in ("multi_select", "yes_no"):
            selected = [o for o in option_scores if o[0] > 0]
            if selected:
                responses.append({
                    "question_id": q_id,
                    "q_type": q_type,
                    "selected_option_ids": [o[1] for o in selected],
                    "selected_option_texts": [o[3] for o in selected],
                    "matched_mappings": [m for o in selected for m in o[2]],
                })

    return responses


def main():
    load_dotenv()

    mongo_url = os.getenv("MONGODB_URL")
    db_name = os.getenv("DB_NAME", "She_Health")
    client = MongoClient(mongo_url)
    db = client[db_name]
    questions = list(db.questions.find({}, {"_id": 0}))

    base = Path(__file__).resolve().parent
    models_dir = base / "app" / "ml" / "models"
    data_dir = base / "dataset" / "final_dataset"

    runs = [
        {
            "name": "PCOS",
            "dataset": data_dir / "pcos.csv",
            "target": "PCOS",
            "artifact": models_dir / "pcos_model.pkl",
            "disease_key": "pcos",
        },
        {
            "name": "Endometriosis",
            "dataset": data_dir / "endometriosis.csv",
            "target": "label",
            "artifact": models_dir / "endometriosis_model.pkl",
            "disease_key": "endo",
        },
        {
            "name": "Cervical",
            "dataset": data_dir / "cervical_cancer.csv",
            "target": "Biopsy",
            "artifact": models_dir / "cervical_cancer_model.pkl",
            "disease_key": "cervical_cancer",
        },
    ]

    forced_thresholds = {
        "PCOS": 0.25,
        "Endometriosis": 0.35,
        "Cervical": 0.10,
    }

    all_results = []

    for run in runs:
        df = read_csv_robust(run["dataset"])
        df.columns = [c.strip() for c in df.columns]

        # match training scripts behavior for cervical data
        if run["name"] == "Cervical":
            df = df.replace(r"\s*\?\s*", pd.NA, regex=True)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # deterministic random sampling
        sample = df.sample(n=5, random_state=42).reset_index(drop=True)

        artifact = joblib.load(run["artifact"])
        model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
        features = artifact.get("features", []) if isinstance(artifact, dict) else []
        threshold = forced_thresholds.get(
            run["name"], artifact.get("threshold", 0.5) if isinstance(artifact, dict) else 0.5
        )

        for i, row in sample.iterrows():
            model_input = build_model_input(row, features)
            x_df = pd.DataFrame([model_input], columns=features).fillna(0)
            prob = get_probability(model, x_df)
            pred = int(prob >= float(threshold))
            actual = int(float(row[run["target"]])) if pd.notna(row[run["target"]]) else 0

            responses = map_questionnaire_responses(row, questions, run["disease_key"])

            all_results.append(
                {
                    "disease": run["name"],
                    "case_id": f"{run['name'][:3].upper()}-{i+1}",
                    "actual": actual,
                    "predicted": pred,
                    "probability": round(prob, 6),
                    "threshold": float(threshold),
                    "questionnaire_response_count": len(responses),
                    "questionnaire_responses_preview": responses[:5],
                    "model_feature_count": len(features),
                    "model_input": model_input,
                }
            )

    out_path = base / "validation_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    summary = {}
    for r in all_results:
        d = r["disease"]
        summary.setdefault(d, {"total": 0, "correct": 0})
        summary[d]["total"] += 1
        if r["actual"] == r["predicted"]:
            summary[d]["correct"] += 1

    print(json.dumps({"summary": summary, "output_file": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()

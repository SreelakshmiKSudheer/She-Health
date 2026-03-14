import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


def norm(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", " ", str(s).strip().lower())
    return " ".join(s.split())


def token_set(s: str):
    return set(norm(s).split())


def best_column_from_feature(feature_name: str, columns):
    base = re.sub(r"_(pcos|endo|endometriosis|cervical_cancer)$", "", feature_name, flags=re.IGNORECASE)
    base_norm = norm(base.replace("_", " "))

    cols_norm = {c: norm(c) for c in columns}
    exact = [c for c, n in cols_norm.items() if n == base_norm]
    if exact:
        return exact[0]

    target_tokens = token_set(base_norm)
    best_col = None
    best_score = 0
    for c, n in cols_norm.items():
        t = token_set(n)
        score = len(target_tokens & t) if t and target_tokens else 0
        if score > best_score:
            best_score = score
            best_col = c
    return best_col if best_score >= 2 else None


def safe_float(v):
    try:
        if pd.isna(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def read_csv_robust(path: Path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def choose_option_ids(q, row, disease_key):
    options = q.get("options") or []
    if not options:
        return []

    cols = list(row.index)
    scored = []
    for opt in options:
        score = 0
        for m in opt.get("mappings", []):
            fname = m.get("feature_name", "")
            # prefer disease-specific mappings if present
            if disease_key and fname.lower().endswith(disease_key):
                pass
            elif any(fname.lower().endswith(s) for s in ["pcos", "endo", "endometriosis", "cervical_cancer"]):
                continue

            col = best_column_from_feature(fname, cols)
            if not col:
                continue
            expected = safe_float(m.get("feature_value", 0.0))
            actual = safe_float(row[col])
            if abs(expected - actual) < 1e-9:
                score += 1
        scored.append((score, opt.get("id")))

    q_type = q.get("q_type")
    if q_type in ("single_select", "yes_no"):
        scored.sort(key=lambda x: x[0], reverse=True)
        top_id = scored[0][1] if scored else options[0].get("id")
        return [top_id]

    if q_type == "multi_select":
        selected = [oid for score, oid in scored if score > 0]
        if selected:
            return selected
        # Keep schema-valid non-empty selection
        return [options[0].get("id")]

    return [options[0].get("id")]


def choose_input_value(q, row, disease_key):
    direct = q.get("direct_mappings") or []
    cols = list(row.index)

    # try disease-specific first
    for m in direct:
        fname = m.get("feature_name", "")
        if disease_key and not fname.lower().endswith(disease_key):
            continue
        col = best_column_from_feature(fname, cols)
        if col:
            return safe_float(row[col])

    # fallback to any mapping
    for m in direct:
        col = best_column_from_feature(m.get("feature_name", ""), cols)
        if col:
            return safe_float(row[col])

    return 0.0


def build_schema_responses(questions, row, disease_key):
    responses = []
    for q in sorted(questions, key=lambda x: x.get("priority", 0)):
        q_id = q.get("id")
        q_type = q.get("q_type")

        if q_type == "input":
            value = choose_input_value(q, row, disease_key)
            selected_option_ids = [f"INPUT::{value}"]
        else:
            selected_option_ids = choose_option_ids(q, row, disease_key)
            if not selected_option_ids:
                # last-resort schema safety
                selected_option_ids = ["INPUT::0"]

        responses.append(
            {
                "question_id": q_id,
                "selected_option_ids": selected_option_ids,
            }
        )
    return responses


def main():
    load_dotenv()
    client = MongoClient(os.getenv("MONGODB_URL"))
    db = client[os.getenv("DB_NAME", "She_Health")]

    users = list(db.users.find({}, {"_id": 0, "user_id": 1}))
    if len(users) < 3:
        raise RuntimeError("Need at least 3 users in DB")

    questions = list(db.questions.find({}, {"_id": 0}))

    base = Path(__file__).resolve().parent
    pcos_df = read_csv_robust(base / "dataset" / "final_dataset" / "pcos.csv")
    pcos_df.columns = [c.strip() for c in pcos_df.columns]
    pcos_row = pcos_df.sample(n=5, random_state=42).reset_index(drop=True).iloc[0]

    endo_df = read_csv_robust(base / "dataset" / "final_dataset" / "endometriosis.csv")
    endo_df.columns = [c.strip() for c in endo_df.columns]
    endo_row = endo_df.sample(n=5, random_state=42).reset_index(drop=True).iloc[1]

    cervical_df = read_csv_robust(base / "dataset" / "final_dataset" / "cervical_cancer.csv")
    cervical_df.columns = [c.strip() for c in cervical_df.columns]
    cervical_df = cervical_df.replace(r"\s*\?\s*", pd.NA, regex=True)
    for c in cervical_df.columns:
        cervical_df[c] = pd.to_numeric(cervical_df[c], errors="coerce")
    cervical_row = cervical_df.sample(n=5, random_state=42).reset_index(drop=True).iloc[3]

    assignments = [
        (users[0]["user_id"], pcos_row, "pcos"),
        (users[1]["user_id"], endo_row, "endo"),
        (users[2]["user_id"], cervical_row, "cervical_cancer"),
    ]

    for user_id, row, disease_key in assignments:
        responses = build_schema_responses(questions, row, disease_key)
        db.user_responses.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "responses": responses,
                    "updated_at": datetime.utcnow(),
                },
                "$setOnInsert": {"created_at": datetime.utcnow()},
            },
            upsert=True,
        )
        print(f"Upserted user_responses for {user_id} with {len(responses)} responses")

    # quick verification output
    for user_id, _, _ in assignments:
        doc = db.user_responses.find_one({"user_id": user_id}, {"_id": 0, "user_id": 1, "responses": 1})
        print(user_id, "stored_responses", len(doc.get("responses", [])))


if __name__ == "__main__":
    main()

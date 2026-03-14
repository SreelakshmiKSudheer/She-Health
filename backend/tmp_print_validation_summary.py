import json

with open("validation_results.json", encoding="utf-8") as f:
    data = json.load(f)

for r in data:
    nz = [(k, v) for k, v in r["model_input"].items() if float(v) != 0.0][:8]
    print(
        "{}|{}|actual={}|pred={}|prob={:.4f}|thr={}|qresp={}|input={}".format(
            r["case_id"],
            r["disease"],
            r["actual"],
            r["predicted"],
            r["probability"],
            r["threshold"],
            r["questionnaire_response_count"],
            nz,
        )
    )

"""
run_baseline.py — the entrypoint that ties it together and records a baseline.

Usage:
  python run_baseline.py --mock              # CPU-only, no GPU (pipeline test)
  python run_baseline.py --vllm              # real: hits vLLM/Qwen endpoint
  python run_baseline.py --vllm --base-url http://<ip>:8000 --model Qwen/Qwen2.5-7B-Instruct

Writes results to results/<label>.json so runs are comparable over time
(this JSON is what a model registry / MLflow would later track).
"""
import argparse
import json
import os
import datetime

from build_db import build, schema_text
from eval_set import EVAL_SET
from harness import evaluate
from predict import vllm_predictor, mock_predictor
from tracking import init_tracking, log_eval_run

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use the CPU mock predictor")
    ap.add_argument("--vllm", action="store_true", help="use the vLLM/OpenAI endpoint")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None, help="label for this run's results file")
    args = ap.parse_args()

    # (re)build the target database so eval is reproducible
    build()

    if args.vllm:
        predict_fn = vllm_predictor(base_url=args.base_url, model=args.model)
        label = args.label or "baseline_vllm"
    else:
        predict_fn = mock_predictor()
        label = args.label or "mock"

    print(f"\nRunning eval  |  predictor={label}\n")
    out = evaluate(EVAL_SET, predict_fn, schema_text(), verbose=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {
        "label": label,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": args.model or os.getenv("LLM_MODEL", "mock"),
        "summary": out["summary"],
        "per_item": [
            {k: v for k, v in item.items() if k in
             ("id", "type", "question", "gold_sql", "pred_sql",
              "executable", "result_match", "note")}
            for item in out["per_item"]
        ],
    }
    path = os.path.join(RESULTS_DIR, f"{label}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved: {path}")

    # log to MLflow so runs are trackable + comparable in the UI
    init_tracking(experiment="text-to-sql")
    log_eval_run(
        label=label,
        summary=out["summary"],
        params={
            "model": args.model or os.getenv("LLM_MODEL", "mock"),
            "predictor": "vllm" if args.vllm else "mock",
            "n_eval_questions": out["summary"]["total"],
        },
        artifact_path=path,
        tags={"phase": label},
    )


if __name__ == "__main__":
    main()

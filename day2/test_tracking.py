"""
test_tracking.py — quick sanity check that MLflow logging works.
Logs two fake runs so you can see them compared in the MLflow UI.
Run:  python test_tracking.py   then:  mlflow ui
"""
from tracking import init_tracking, log_eval_run

init_tracking(experiment="text-to-sql-test")

# fake "baseline" run
log_eval_run(
    label="fake_baseline",
    summary={"execution_accuracy": 0.80, "result_match_accuracy": 0.53},
    params={"model": "Qwen/Qwen2.5-7B-Instruct", "phase": "baseline", "n_examples": 0},
    tags={"phase": "baseline"},
)

# fake "fine-tuned" run — pretend it improved
log_eval_run(
    label="fake_finetune_r16",
    summary={"execution_accuracy": 0.93, "result_match_accuracy": 0.71},
    params={"model": "qwen-sql-lora", "phase": "finetune", "rank": 16, "n_examples": 5000, "lr": 2e-4},
    tags={"phase": "finetune"},
)

print("\nDone. Now run:  mlflow ui   and open http://localhost:5000")

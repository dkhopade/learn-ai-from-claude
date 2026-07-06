"""
tracking.py — thin MLflow wrapper for logging eval + fine-tune runs.

MLflow concepts used here:
  - experiment : a named grouping of related runs (e.g. "text-to-sql")
  - run        : one execution — has params, metrics, and artifacts
  - params     : inputs you set (model, rank, lr, data size) — logged once
  - metrics    : outputs you measure (accuracy) — can be logged over time
  - artifacts  : files attached to the run (results JSON, plots, adapters)

Runs are stored locally under ./mlruns by default (no server needed).
Launch the UI with:  mlflow ui   (then open http://localhost:5000)
"""
import os
import mlflow


DEFAULT_EXPERIMENT = "text-to-sql"


def init_tracking(experiment: str = DEFAULT_EXPERIMENT, tracking_uri: str = None):
    """
    Point MLflow at a tracking location and select the experiment.

    tracking_uri:
      - None  -> local ./mlruns folder (default, zero setup)
      - a URL -> a remote MLflow server (e.g. the one we'll run on OKE later)
    """
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return tracking_uri


def log_eval_run(label: str, summary: dict, params: dict = None,
                 artifact_path: str = None, tags: dict = None):
    """
    Log a single evaluation run.

    label        : run name (e.g. "baseline_qwen", "finetune_r16_5k")
    summary      : the metrics dict from the harness
                   (execution_accuracy, result_match_accuracy, ...)
    params       : config for this run (model, rank, lr, n_examples, ...)
    artifact_path: optional path to a file to attach (e.g. results/xxx.json)
    tags         : optional key/value tags (e.g. {"phase": "baseline"})
    """
    with mlflow.start_run(run_name=label):
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        # log the numeric metrics
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # attach the results file if provided
        if artifact_path and os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)

        run_id = mlflow.active_run().info.run_id
    print(f"  logged run '{label}' (id={run_id[:8]}) to experiment")
    return run_id

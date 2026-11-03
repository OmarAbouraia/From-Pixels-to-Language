"""Experiment tracking with MLflow.

Every training run in this project already leaves a folder with:
    training_history.csv     per-epoch train/val loss and accuracy
    evaluation_metrics.json  final metrics + checkpoint info (from `cli evaluate`)
    trial_result.json        tuning trials only: lr, wd, training time
    best_model.pth           checkpoint (NOT uploaded; too large, referenced by path)

`log_run_folder` turns one such folder into one MLflow run:
    params   - model, mode, learning rate, weight decay, batch size, ...
    metrics  - final validation metrics, plus the per-epoch curves (step = epoch)
    tags     - checkpoint path and source folder
    artifacts- the small CSV/JSON/PNG files of the run

`log_runs` scans a directory tree and logs every run folder it finds, so
runs made before tracking existed (Feature Extraction, Fine-Tuning, WD-*,
Grid/Random Search trials) are tracked without retraining. A folder that is
already logged is skipped, so the command can be run again safely.

Storage: one SQLite file (mlflow.db) + an artifacts/ folder inside
tracking_dir. SQLite is built in a local temp copy and copied back at the end,
because SQLite can lock up when written directly on Google Drive.
"""

import csv
import json
import shutil
import tempfile
from pathlib import Path

PARAM_KEYS = {"model_name", "mode", "learning_rate", "weight_decay", "batch_size",
              "epoch", "best_epoch", "method", "trial", "pretrained", "optimizer",
              "seed", "epochs"}
SMALL_ARTIFACTS = {".csv", ".json", ".png", ".yaml", ".txt"}
DB_NAME = "mlflow.db"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)}
                for row in csv.DictReader(file)]


def collect_run_info(run_dir: Path) -> dict:
    """Read params, final metrics and curves from a run folder (no MLflow needed)."""
    run_dir = Path(run_dir)
    evaluation = _read_json(run_dir / "evaluation_metrics.json")
    trial = _read_json(run_dir / "trial_result.json")
    history = read_history(run_dir / "training_history.csv")
    merged = {**evaluation, **trial}

    params = {"model_name": "resnet18", "pretrained": True, "optimizer": "adam"}
    metrics = {}
    for key, value in merged.items():
        if key in PARAM_KEYS:
            params[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = float(value)
    if "epoch" in params:                      # checkpoint epoch = best epoch
        params["best_epoch"] = params.pop("epoch")
    if history:
        params["epochs_trained"] = int(history[-1]["epoch"])
    if "mode" not in params:                   # infer from folder name if missing
        name = str(run_dir).lower()
        params["mode"] = ("feature_extraction" if "feature" in name
                          else "fine_tuning")

    checkpoint = run_dir / "best_model.pth"
    return {
        "params": params,
        "metrics": metrics,
        "history": history,
        "checkpoint": str(checkpoint) if checkpoint.exists()
                      else evaluation.get("checkpoint_path", ""),
        "evaluated": bool(evaluation),
    }


def find_run_folders(root: Path, exclude: Path | None = None) -> list[Path]:
    """Every folder under root that contains a training_history.csv.

    `exclude` (the tracking folder) is skipped: MLflow stores copies of each
    run's training_history.csv in its artifacts, and those must not be
    mistaken for new runs when the tracking folder sits inside `root`.
    """
    excluded = Path(exclude).resolve() if exclude is not None else None
    folders = []
    for history in Path(root).rglob("training_history.csv"):
        folder = history.parent.resolve()
        if excluded is not None and (folder == excluded or excluded in folder.parents):
            continue
        folders.append(history.parent)
    return sorted(folders)


def log_runs(root, tracking_dir, experiment="classification", run_name_prefix=""):
    """Log every run folder under `root` to MLflow. Returns the logged run names."""
    import mlflow

    root, tracking_dir = Path(root), Path(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = (tracking_dir / "artifacts").resolve()

    with tempfile.TemporaryDirectory() as tmp:
        local_db = Path(tmp) / DB_NAME
        if (tracking_dir / DB_NAME).exists():
            shutil.copy2(tracking_dir / DB_NAME, local_db)
        mlflow.set_tracking_uri(f"sqlite:///{local_db}")

        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment,
                                     artifact_location=artifact_root.as_uri())
        mlflow.set_experiment(experiment)

        logged = []
        for run_dir in find_run_folders(root, exclude=tracking_dir):
            source = str(run_dir.resolve())
            existing = mlflow.search_runs(
                filter_string=f"tags.source_dir = '{source}'", max_results=1)
            if len(existing):
                print(f"already tracked, skipping: {run_dir}")
                continue

            info = collect_run_info(run_dir)
            relative = run_dir.relative_to(root).as_posix() if run_dir != root \
                else run_dir.name
            run_name = f"{run_name_prefix}{relative}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(info["params"])
                mlflow.log_metrics(info["metrics"])
                for row in info["history"]:
                    step = int(row["epoch"])
                    mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"},
                                       step=step)
                mlflow.set_tags({"source_dir": source,
                                 "checkpoint_path": info["checkpoint"],
                                 "evaluated": str(info["evaluated"]).lower(),
                                 "task": "classification"})
                for file in run_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in SMALL_ARTIFACTS:
                        mlflow.log_artifact(str(file))
            logged.append(run_name)
            print(f"tracked: {run_name}")

        shutil.copy2(local_db, tracking_dir / DB_NAME)

    print(f"\n{len(logged)} new run(s) logged to {tracking_dir / DB_NAME}")
    return logged


def export_summary(tracking_dir, experiment="classification", output_dir=None):
    """Write runs_summary.csv and a plot of every run's validation curves.

    Gives report-ready tables/figures without opening the MLflow web UI.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    tracking_dir = Path(tracking_dir)
    output_dir = Path(output_dir or tracking_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking_dir / DB_NAME).resolve()}")
    client = mlflow.tracking.MlflowClient()
    experiment_obj = client.get_experiment_by_name(experiment)
    runs = client.search_runs([experiment_obj.experiment_id],
                              order_by=["attributes.run_name ASC"])

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for run in runs:
        name = run.info.run_name
        row = {"run": name, **run.data.params,
               **{k: round(v, 4) for k, v in run.data.metrics.items()
                  if k in ("accuracy", "top5_accuracy", "macro_f1", "val_loss",
                           "train_seconds")}}
        rows.append(row)
        for axis, key in zip(axes, ("val_loss", "val_accuracy")):
            curve = client.get_metric_history(run.info.run_id, key)
            if len(curve) > 1:
                curve = sorted(curve, key=lambda m: m.step)
                axis.plot([m.step for m in curve], [m.value for m in curve],
                          marker=".", label=name)
    for axis, title in zip(axes, ("Validation loss", "Validation accuracy")):
        axis.set(xlabel="Epoch", title=f"{title} (all tracked runs)")
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "tracked_runs_curves.png", dpi=120)
    plt.close(fig)

    fields = sorted({k for row in rows for k in row}, key=lambda k: (k != "run", k))
    with (output_dir / "runs_summary.csv").open("w", newline="",
                                                encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} runs to {output_dir}")
    return rows

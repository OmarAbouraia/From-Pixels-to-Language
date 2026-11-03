"""Grid Search vs Random Search for the pretrained ResNet18 (Milestone 1).

Tuned hyperparameters: learning rate and weight decay only. Everything else
(model, mode, split, transforms, batch size, epochs, seed, metric) is fixed,
so the two methods are compared fairly.

  Grid Search   tries every combination of a few hand-picked values.
  Random Search samples the same number of trials from continuous ranges
                (log-uniform, because good learning rates span orders of
                magnitude).

Each trial = one call to the existing src.transfer_learning.train.run_experiment
(no training code is duplicated), then the best checkpoint of that trial is
evaluated on the 1,198-image validation split. Selection metric: macro F1.

Colab sessions disconnect, so every finished trial writes trial_result.json;
re-running the same command skips finished trials and continues.
"""

import csv
import itertools
import json
import math
import random
import time
from pathlib import Path

from src.utils import set_seed

RESULT_FILE = "trial_result.json"
RESULT_FIELDS = ["method", "trial", "learning_rate", "weight_decay", "best_epoch",
                 "val_loss", "accuracy", "top5_accuracy", "macro_precision",
                 "macro_recall", "macro_f1", "train_seconds", "trial_dir"]


# ---------------------------------------------------------------------------
# search spaces
# ---------------------------------------------------------------------------

def grid_trials(grid: dict) -> list[dict]:
    """Every (learning_rate, weight_decay) combination, in a fixed order."""
    return [{"learning_rate": float(lr), "weight_decay": float(wd)}
            for lr, wd in itertools.product(grid["learning_rate"],
                                            grid["weight_decay"])]


def _sample(rng: random.Random, space: dict) -> float:
    low, high = float(space["low"]), float(space["high"])
    if space.get("log", True):
        value = 10 ** rng.uniform(math.log10(low), math.log10(high))
    else:
        value = rng.uniform(low, high)
    return float(f"{value:.3g}")        # 3 significant digits, easier to read


def random_trials(random_config: dict, n_trials: int, seed: int) -> list[dict]:
    """n_trials samples; the same seed always gives the same trials."""
    rng = random.Random(seed)
    return [{"learning_rate": _sample(rng, random_config["learning_rate"]),
             "weight_decay": _sample(rng, random_config["weight_decay"])}
            for _ in range(n_trials)]


# ---------------------------------------------------------------------------
# one trial
# ---------------------------------------------------------------------------

def _trial_name(index: int, params: dict) -> str:
    return (f"trial_{index:02d}_lr{params['learning_rate']:.0e}"
            f"_wd{params['weight_decay']:.0e}")


def run_trial(method, index, params, config, val_loader, class_names, device):
    """Train, evaluate and record one configuration (or reuse a finished one)."""
    from src.evaluation import evaluate_model
    from src.models import load_classifier
    from src.transfer_learning.train import run_experiment

    trial_dir = Path(config["output_dir"]) / method / _trial_name(index, params)
    result_path = trial_dir / RESULT_FILE
    if result_path.exists():
        print(f"[{method} {index}] already finished, skipping: {trial_dir.name}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    print(f"\n[{method} {index}] lr={params['learning_rate']:g} "
          f"wd={params['weight_decay']:g}")
    set_seed(int(config.get("seed", 42)))
    start = time.perf_counter()
    run_experiment(
        mode=config.get("mode", "fine_tuning"),
        epochs=int(config["epochs"]),
        batch_size=int(config.get("batch_size", 32)),
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        num_workers=int(config.get("num_workers", 0)),
        output_dir=trial_dir,
    )
    train_seconds = time.perf_counter() - start

    model, meta = load_classifier(trial_dir / "best_model.pth",
                                  name=config.get("model", "resnet18"),
                                  device=device)
    metrics = evaluate_model(model, val_loader, class_names, output_dir=trial_dir,
                             device=device)

    result = {"method": method, "trial": index, **params,
              "best_epoch": meta.get("epoch"), "train_seconds": round(train_seconds, 1),
              "trial_dir": str(trial_dir),
              **{k: metrics[k] for k in RESULT_FIELDS if k in metrics}}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Checkpoints are large (~129 MB each). last_checkpoint.pth is never
    # needed for tuning; best_model.pth is kept only if asked.
    (trial_dir / "last_checkpoint.pth").unlink(missing_ok=True)
    if not config.get("keep_best_checkpoints", True):
        (trial_dir / "best_model.pth").unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def summarize(method: str, results: list[dict], metric: str = "macro_f1") -> dict:
    """Best score, trial count and time for one method.

    trials_to_best / seconds_to_best: how much budget was spent before the
    best trial was found (lower = the method found good settings faster).
    """
    best_position = max(range(len(results)), key=lambda i: results[i][metric])
    best = results[best_position]
    seconds = [r["train_seconds"] for r in results]
    return {
        "method": method,
        "num_trials": len(results),
        f"best_{metric}": best[metric],
        "best_accuracy": best["accuracy"],
        "best_val_loss": best["val_loss"],
        "best_learning_rate": best["learning_rate"],
        "best_weight_decay": best["weight_decay"],
        "trials_to_best": best_position + 1,
        "total_seconds": round(sum(seconds), 1),
        "seconds_to_best": round(sum(seconds[:best_position + 1]), 1),
        f"mean_{metric}": sum(r[metric] for r in results) / len(results),
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(all_results: dict, output_path: Path, metric="macro_f1"):
    """Best-so-far score vs trial number and vs time, plus the sampled points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401  (used for log-axis labels)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for method, results in all_results.items():
        scores = [r[metric] * 100 for r in results]
        best_so_far = [max(scores[:i + 1]) for i in range(len(scores))]
        minutes = list(itertools.accumulate(r["train_seconds"] / 60 for r in results))
        axes[0].plot(range(1, len(scores) + 1), best_so_far, marker="o", label=method)
        axes[1].plot(minutes, best_so_far, marker="o", label=method)
        points = axes[2].scatter([r["learning_rate"] for r in results],
                                 [max(r["weight_decay"], 1e-7) for r in results],
                                 c=scores, cmap="viridis", s=80,
                                 marker="s" if method == "grid" else "o",
                                 label=method, edgecolors="black")
    axes[0].set(xlabel="Trial", ylabel=f"Best {metric} so far (%)",
                title="Score vs number of trials")
    axes[1].set(xlabel="Cumulative training time (min)",
                ylabel=f"Best {metric} so far (%)", title="Score vs time")
    axes[2].set(xscale="log", yscale="log", xlabel="Learning rate",
                ylabel="Weight decay (0 shown as 1e-7)",
                title="Tried configurations (colour = score)")
    axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.colorbar(points, ax=axes[2], label=f"{metric} (%)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run_tuning(config: dict, methods: str = "both", device=None):
    """Run grid and/or random search and write the comparison files."""
    import torch

    from src.cli import get_val_loader
    from src.inference import load_class_names

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = config.get("selection_metric", "macro_f1")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = grid_trials(config["grid"])
    n_random = int(config["random"].get("n_trials") or len(grid))  # same budget
    plans = {
        "grid": grid,
        "random": random_trials(config["random"], n_random,
                                int(config["random"].get("seed", 0))),
    }
    selected = ["grid", "random"] if methods == "both" else [methods]

    print(f"Tuning {', '.join(selected)} | {config['epochs']} epochs per trial | "
          f"selection metric: {metric}")
    for name in selected:
        print(f"  {name}: {len(plans[name])} trials")

    val_loader = get_val_loader(config.get("model", "resnet18"),
                                int(config.get("batch_size", 32)),
                                int(config.get("num_workers", 0)))
    class_names = load_class_names(config.get("class_names_file"))

    all_results = {}
    for name in selected:
        all_results[name] = [
            run_trial(name, index, params, config, val_loader, class_names, device)
            for index, params in enumerate(plans[name], start=1)
        ]
        _write_csv(output_dir / f"{name}_results.csv", all_results[name], RESULT_FIELDS)

    # Summaries use every finished method found on disk, so running
    # --method grid and --method random on different days still compares both.
    for name in ("grid", "random"):
        if name not in all_results and (output_dir / name).exists():
            finished = sorted((output_dir / name).glob(f"*/{RESULT_FILE}"))
            if finished:
                all_results[name] = sorted(
                    (json.loads(p.read_text(encoding="utf-8")) for p in finished),
                    key=lambda r: r["trial"])

    summaries = [summarize(name, results, metric)
                 for name, results in all_results.items()]
    _write_csv(output_dir / "comparison.csv", summaries, list(summaries[0].keys()))
    (output_dir / "comparison.json").write_text(
        json.dumps({"config": config, "summaries": summaries}, indent=2, default=str),
        encoding="utf-8")
    plot_comparison(all_results, output_dir / "tuning_comparison.png", metric)

    print("\n" + "=" * 60 + "\nTUNING COMPARISON\n" + "=" * 60)
    for s in summaries:
        print(f"{s['method']:<7} trials={s['num_trials']:<3} "
              f"best {metric}={s[f'best_{metric}'] * 100:.2f}% "
              f"(lr={s['best_learning_rate']:g}, wd={s['best_weight_decay']:g}) "
              f"found at trial {s['trials_to_best']} | "
              f"total {s['total_seconds'] / 60:.1f} min")
    print(f"\nFiles saved in {output_dir}")
    return summaries

"""Project command line interface.

    python -m src.cli train   --task classification --config configs/classification_fine_tuning.yaml
    python -m src.cli predict --task classification --checkpoint best_model.pth --image bird.jpg
    python -m src.cli evaluate --task classification --checkpoint best_model.pth --output-dir outputs/eval
    python -m src.cli gradcam --task classification --checkpoint best_model.pth --output-dir outputs/gradcam
    python -m src.cli tune    --task classification --config configs/tuning_fine_tuning.yaml --method both

Design choice: this file only parses arguments and dispatches. Training is
delegated to the existing src.transfer_learning.train.run_experiment (not
re-implemented), and model building/loading goes through src.models.
Only the classification task exists for now; colorization and captioning
add their own branches in Milestone 2.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from src.utils import set_seed

TASKS = ["classification"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must be a YAML mapping.")
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_val_loader(model_name: str, batch_size: int = 32, num_workers: int = 0):
    """Validation loader of the original EDA split (1,198 images, not shuffled),
    with the preprocessing that matches the model."""
    from src.models import get_model_spec

    if get_model_spec(model_name).normalization == "imagenet":
        from src.transfer_learning.data import create_transfer_learning_loaders

        _, val_loader = create_transfer_learning_loaders(
            batch_size=batch_size, num_workers=num_workers)
    else:
        from src.data.data_loader import create_data_loaders

        _, val_loader = create_data_loaders(
            batch_size=batch_size, num_workers=num_workers)
    return val_loader


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    config = load_config(args.config)
    # Command-line flags override the config file.
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay",
                "output_dir", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    model_name = config.get("model", "resnet18")
    set_seed(int(config.get("seed", 42)))
    print(f"Config: {args.config}")
    print(json.dumps(config, indent=2))

    if model_name == "resnet18":
        from src.transfer_learning.train import run_experiment

        run_experiment(
            mode=config.get("mode", "fine_tuning"),
            epochs=int(config.get("epochs", 10)),
            batch_size=int(config.get("batch_size", 32)),
            learning_rate=config.get("learning_rate"),
            weight_decay=float(config.get("weight_decay", 0.0)),
            num_workers=int(config.get("num_workers", 0)),
            output_dir=config.get("output_dir"),
        )
        return 0

    print(f"ERROR: training for model '{model_name}' is not connected yet. "
          "Register the model in src/models/registry.py and add its training "
          "entry point here.", file=sys.stderr)
    return 2


def cmd_predict(args) -> int:
    from src.inference import load_class_names, predict_image
    from src.models import load_classifier

    device = resolve_device(args.device)
    model, meta = load_classifier(args.checkpoint, name=args.model, device=device)
    class_names = load_class_names(args.class_names)
    results = predict_image(model, args.model, args.image, class_names, args.top_k)

    print(f"Model: {args.model} ({meta.get('mode', 'unknown mode')}), "
          f"checkpoint epoch {meta.get('epoch', '?')}")
    print(f"Image: {args.image}\n")
    for row in results:
        print(f"{row['rank']}. {row['species']:<35} {row['probability'] * 100:6.2f}%")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.json}")
    return 0


def cmd_evaluate(args) -> int:
    from src.evaluation import evaluate_model, print_metrics
    from src.inference import load_class_names
    from src.models import load_classifier

    device = resolve_device(args.device)
    model, meta = load_classifier(args.checkpoint, name=args.model, device=device)
    class_names = load_class_names(args.class_names)
    val_loader = get_val_loader(args.model, args.batch_size, args.num_workers)

    metrics = evaluate_model(model, val_loader, class_names,
                             output_dir=args.output_dir, device=device,
                             extra_info=meta)
    print_metrics(metrics)
    return 0


def cmd_tune(args) -> int:
    from src.tuning import run_tuning

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    run_tuning(config, methods=args.method)
    return 0


def cmd_gradcam(args) -> int:
    from src.inference import load_class_names
    from src.models import load_classifier
    from src.visualization.gradcam_examples import generate_gradcam_examples

    device = resolve_device(args.device)
    model, _ = load_classifier(args.checkpoint, name=args.model, device=device)
    class_names = load_class_names(args.class_names)

    val_loader = get_val_loader(args.model, args.batch_size, args.num_workers)
    generate_gradcam_examples(
        model, args.model, val_loader, class_names, args.output_dir,
        num_correct=args.num_correct, num_incorrect=args.num_incorrect,
        device=device,
    )
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli",
                                     description="From Pixels to Language CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_args(p):
        p.add_argument("--task", choices=TASKS, default="classification")
        p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
        p.add_argument("--model", default="resnet18", help="Registered model name")
        p.add_argument("--class-names", default=None,
                       help="Optional JSON list of species names (works offline)")
        p.add_argument("--device", default="auto", help="auto, cpu or cuda")

    train = sub.add_parser("train", help="Train a model from a YAML config")
    train.add_argument("--task", choices=TASKS, default="classification")
    train.add_argument("--config", required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", dest="batch_size", type=int)
    train.add_argument("--learning-rate", dest="learning_rate", type=float)
    train.add_argument("--weight-decay", dest="weight_decay", type=float)
    train.add_argument("--output-dir", dest="output_dir")
    train.add_argument("--seed", type=int)
    train.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="Predict the species of one image")
    add_model_args(predict)
    predict.add_argument("--image", required=True)
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--json", default=None, help="Also save results as JSON")
    predict.set_defaults(func=cmd_predict)

    evaluate = sub.add_parser("evaluate",
                              help="Metrics, confusion matrix and predictions on the validation split")
    add_model_args(evaluate)
    evaluate.add_argument("--output-dir", default=None,
                          help="Where to save evaluation_metrics.json etc. "
                               "(default: print only)")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.set_defaults(func=cmd_evaluate)

    tune = sub.add_parser("tune", help="Grid Search and/or Random Search")
    tune.add_argument("--task", choices=TASKS, default="classification")
    tune.add_argument("--config", required=True)
    tune.add_argument("--method", choices=["grid", "random", "both"], default="both")
    tune.add_argument("--output-dir", default=None)
    tune.set_defaults(func=cmd_tune)

    gradcam = sub.add_parser("gradcam",
                             help="Grad-CAM for correct/incorrect validation images")
    add_model_args(gradcam)
    gradcam.add_argument("--output-dir", default="outputs/gradcam")
    gradcam.add_argument("--num-correct", type=int, default=8)
    gradcam.add_argument("--num-incorrect", type=int, default=8)
    gradcam.add_argument("--batch-size", type=int, default=32)
    gradcam.add_argument("--num-workers", type=int, default=0)
    gradcam.set_defaults(func=cmd_gradcam)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

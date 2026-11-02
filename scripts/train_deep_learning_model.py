"""Train the Deep Learning add-on and create a same-split model comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deep_learning import TrainingConfig, train_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/raw/telco_customer_churn.csv",
        help="Path to the labeled Telco Customer Churn CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/deep_learning",
        help="Directory for Deep Learning artifacts and comparison metrics.",
    )
    parser.add_argument(
        "--text-features",
        default=None,
        help="Optional customer-level NLP feature CSV from the NLP pipeline.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-rf-baseline",
        action="store_true",
        help="Skip the calibrated Random Forest comparator (not recommended for final evaluation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = (ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    text_features = None
    if args.text_features:
        text_features = (ROOT / args.text_features).resolve() if not Path(args.text_features).is_absolute() else Path(args.text_features)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found: {data_path}\n"
            "Place the Telco Customer Churn CSV at data/raw/telco_customer_churn.csv, "
            "or pass --data PATH."
        )
    config = TrainingConfig(
        max_epochs=max(1, args.epochs),
        patience=max(1, args.patience),
        batch_size=max(1, args.batch_size),
        learning_rate=args.learning_rate,
        threshold=args.threshold,
    )
    comparison = train_from_csv(
        data_path=data_path,
        output_dir=output_dir,
        text_features_path=text_features,
        config=config,
        include_rf_baseline=not args.skip_rf_baseline,
    )
    print(f"Add-on artifacts saved under: {output_dir}")
    print(f"Comparison metrics: {output_dir / 'comparison_metrics.json'}")
    for model_name, metrics in comparison["metrics"].items():
        print(f"\n{model_name}")
        for metric_name in ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier_score"):
            print(f"  {metric_name}: {metrics[metric_name]:.4f}")
    if text_features:
        print("\nWARNING: Check that every support ticket used to build NLP features predates the churn prediction cutoff.")


if __name__ == "__main__":
    main()

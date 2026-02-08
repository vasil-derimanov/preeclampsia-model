"""Entry point — runs the full preeclampsia prediction pipeline."""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, FEATURE_SETS, TARGETS
from src.data import generate_synthetic_data, load_data, preprocess, split_data
from src.evaluation import compute_metrics, plot_roc_curves
from src.models import get_all_models
from src.pipeline import run_repeated_cv, summarize_cv_results


def parse_args():
    parser = argparse.ArgumentParser(description="Preeclampsia Prediction Model")
    parser.add_argument(
        "--data", type=str, default=None, help="Path to CSV dataset"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data for demo"
    )
    parser.add_argument(
        "--n-samples", type=int, default=4644, help="Number of synthetic samples"
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5, help="Number of CV repetitions (paper uses 200)"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["pe_all", "preterm_pe"],
        choices=list(TARGETS.keys()),
        help="Which targets to evaluate",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["full"],
        choices=list(FEATURE_SETS.keys()),
        help="Which feature set combinations to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Which models to evaluate (default: all). Choices: LR, ETC, VC, GPC, SC",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory for results"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load or generate data ---
    if args.synthetic or args.data is None:
        print(f"Generating synthetic data (n={args.n_samples})...")
        df = generate_synthetic_data(n_samples=args.n_samples)
    else:
        print(f"Loading data from {args.data}...")
        df = load_data(args.data)

    # --- Preprocess ---
    print("Preprocessing...")
    df, scaler = preprocess(df)

    # --- Create output dir ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Get models ---
    all_models = get_all_models()
    model_names = args.models or list(all_models.keys())
    model_names = [m.upper() for m in model_names]

    # --- Run evaluation ---
    for target_name in args.targets:
        print(f"\n{'='*60}")
        print(f"Target: {target_name}")
        print(f"{'='*60}")

        y = df[target_name].values
        print(f"  Positive cases: {y.sum()} / {len(y)} ({100*y.mean():.1f}%)")

        for fs_name in args.feature_sets:
            feature_cols = FEATURE_SETS[fs_name]
            available_cols = [c for c in feature_cols if c in df.columns]
            X = df[available_cols].values

            print(f"\n  Feature set: {fs_name} ({len(available_cols)} features)")
            print(f"  {'-'*50}")

            roc_data = {}
            summary_rows = []

            for model_name in model_names:
                if model_name not in all_models:
                    print(f"  WARNING: Unknown model '{model_name}', skipping.")
                    continue

                factory_fn, _ = all_models[model_name]
                model = factory_fn()

                print(f"  Running {model_name} ({args.n_repeats}x5-fold CV)...", end=" ", flush=True)
                t0 = time.time()

                cv_results = run_repeated_cv(
                    model, X, y, n_repeats=args.n_repeats, n_folds=5
                )

                elapsed = time.time() - t0

                if cv_results.empty:
                    print(f"SKIPPED (not enough positive cases for CV) [{elapsed:.1f}s]")
                    continue

                summary = summarize_cv_results(cv_results)
                auc_row = summary[summary["metric"] == "auc"].iloc[0]
                print(
                    f"AUC={auc_row['mean']:.3f} "
                    f"({auc_row['ci_lower']:.3f}, {auc_row['ci_upper']:.3f}) "
                    f"[{elapsed:.1f}s]"
                )

                # Store for ROC plot (use a single train-test split)
                X_train, X_test, y_train, y_test = split_data(
                    pd.DataFrame(X), pd.Series(y)
                )
                model_fitted = factory_fn()
                try:
                    model_fitted.fit(X_train.values, y_train.values)
                    y_prob = model_fitted.predict_proba(X_test.values)[:, 1]
                    roc_data[model_name] = (y_test.values, y_prob)
                except ValueError:
                    pass  # skip ROC for this model if fitting fails

                # Collect summary
                for _, row in summary.iterrows():
                    summary_rows.append(
                        {
                            "model": model_name,
                            "metric": row["metric"],
                            "mean": row["mean"],
                            "ci_lower": row["ci_lower"],
                            "ci_upper": row["ci_upper"],
                        }
                    )

            # Save summary table
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                csv_path = os.path.join(
                    args.output_dir, f"{target_name}_{fs_name}_results.csv"
                )
                summary_df.to_csv(csv_path, index=False)
                print(f"\n  Results saved to {csv_path}")

            # Plot ROC curves
            if roc_data:
                roc_path = os.path.join(
                    args.output_dir, f"{target_name}_{fs_name}_roc.png"
                )
                plot_roc_curves(
                    roc_data,
                    title=f"ROC - {target_name} ({fs_name})",
                    save_path=roc_path,
                )

    print(f"\nDone. Results in '{args.output_dir}/'")


if __name__ == "__main__":
    main()

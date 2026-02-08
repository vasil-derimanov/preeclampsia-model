"""Training/validation orchestration: repeated CV, bootstrap CI, hyperparameter tuning."""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from skopt import BayesSearchCV

from src.evaluation import compute_calibration, compute_metrics


def run_repeated_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 200,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Run repeated stratified k-fold CV, collecting per-fold metrics.

    Args:
        model: sklearn estimator (unfitted; will be cloned per fold)
        X: feature matrix
        y: target vector
        n_repeats: number of CV repetitions
        n_folds: number of folds per repetition

    Returns:
        DataFrame with one row per fold, columns = metric names.
    """
    from sklearn.base import clone

    rskf = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=42
    )

    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip folds where train or test set has only one class
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_train, y_train)
            except ValueError:
                # Ensemble models (e.g. StackingClassifier) may fail internally
                # when their own CV folds end up with a single class
                continue
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_prob)
        cal = compute_calibration(y_test, y_prob)
        metrics.update(cal)
        metrics["fold"] = fold_idx
        records.append(metrics)

    return pd.DataFrame(records)


def bootstrap_ci(
    values: np.ndarray, alpha: float = 0.05, n_bootstrap: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.array(
        [rng.choice(values, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), float(lower), float(upper)


def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated CV results with bootstrap 95% CIs.

    Returns DataFrame with columns: metric, mean, ci_lower, ci_upper.
    """
    metric_cols = [c for c in cv_results.columns if c != "fold"]
    rows = []
    for col in metric_cols:
        mean, lo, hi = bootstrap_ci(cv_results[col].values)
        rows.append({"metric": col, "mean": mean, "ci_lower": lo, "ci_upper": hi})
    return pd.DataFrame(rows)


def tune_hyperparameters(
    model,
    search_space: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 30,
    cv_splits: int = 5,
    cv_repeats: int = 20,
    seed: int = 42,
):
    """Bayesian hyperparameter optimization using scikit-optimize.

    Uses repeated stratified k-fold CV with AUC-ROC as the scoring metric.

    Returns:
        The best estimator found.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits, n_repeats=cv_repeats, random_state=seed
    )
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    opt.fit(X, y)
    print(f"Best params: {opt.best_params_}")
    print(f"Best AUC-ROC: {opt.best_score_:.4f}")
    return opt.best_estimator_

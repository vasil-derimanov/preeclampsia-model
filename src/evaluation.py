"""Metrics computation, ROC plots, calibration assessment, and SHAP analysis."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute discrimination metrics.

    Returns dict with: auc, sensitivity, specificity (Youden-index cutoff),
    detection rate at 10% FPR, detection rate at 20% FPR.
    """
    if len(np.unique(y_true)) < 2:
        return {
            "auc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "dr_10fpr": float("nan"),
            "dr_20fpr": float("nan"),
        }

    auc = roc_auc_score(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Youden index optimal cutoff
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]

    # DR at fixed FPR thresholds
    dr_10fpr = _dr_at_fpr(fpr, tpr, target_fpr=0.10)
    dr_20fpr = _dr_at_fpr(fpr, tpr, target_fpr=0.20)

    return {
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dr_10fpr": dr_10fpr,
        "dr_20fpr": dr_20fpr,
    }


def _dr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Interpolate the detection rate (TPR) at a given FPR."""
    return float(np.interp(target_fpr, fpr, tpr))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute calibration metrics: Brier score, calibration slope & intercept."""
    brier = brier_score_loss(y_true, y_prob)

    # Need at least 2 classes for logistic recalibration
    if len(np.unique(y_true)) < 2:
        return {
            "brier_score": brier,
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
        }

    # Calibration slope and intercept via logistic recalibration
    log_odds = np.log(np.clip(y_prob, 1e-10, 1 - 1e-10) / (1 - np.clip(y_prob, 1e-10, 1 - 1e-10)))
    cal_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal_model.fit(log_odds.reshape(-1, 1), y_true)

    return {
        "brier_score": brier,
        "calibration_slope": float(cal_model.coef_[0, 0]),
        "calibration_intercept": float(cal_model.intercept_[0]),
    }


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------


def plot_roc_curves(results: dict, title: str = "ROC Curves", save_path: str | None = None):
    """Plot multi-model ROC curves.

    Args:
        results: dict of {model_name: (y_true, y_prob)}
        title: plot title
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_true, y_prob) in results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"ROC plot saved to {save_path}")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------


def plot_shap_analysis(model, X, feature_names: list[str] | None = None, save_path: str | None = None):
    """Generate SHAP beeswarm and pie chart for a fitted model.

    Args:
        model: fitted sklearn estimator
        X: feature matrix (numpy array or DataFrame)
        feature_names: list of feature names
        save_path: base path for saving (will append _beeswarm.png / _pie.png)
    """
    import shap

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Beeswarm plot
    fig_bee, ax_bee = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    if save_path:
        plt.savefig(f"{save_path}_beeswarm.png", dpi=150, bbox_inches="tight")
        print(f"SHAP beeswarm saved to {save_path}_beeswarm.png")
    plt.close()

    # Pie chart of mean absolute SHAP values
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(mean_abs))]
    contributions = dict(zip(feature_names, mean_abs))

    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(
        contributions.values(),
        labels=contributions.keys(),
        autopct="%1.1f%%",
        startangle=140,
    )
    ax_pie.set_title("Feature Contribution (Mean |SHAP|)")
    fig_pie.tight_layout()

    if save_path:
        fig_pie.savefig(f"{save_path}_pie.png", dpi=150, bbox_inches="tight")
        print(f"SHAP pie chart saved to {save_path}_pie.png")
    plt.close(fig_pie)

    return contributions

"""Five classifier definitions matching the paper, with Bayesian search spaces."""

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skopt.space import Categorical, Integer, Real


# ---------------------------------------------------------------------------
# Factory functions — each returns a configured estimator
# ---------------------------------------------------------------------------


def create_logistic_regression(**kwargs) -> LogisticRegression:
    """Logistic Regression with balanced class weight."""
    defaults = dict(class_weight="balanced", max_iter=1000, solver="lbfgs")
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def create_extra_trees(**kwargs) -> ExtraTreesClassifier:
    """Extra Trees Classifier with balanced class weight."""
    defaults = dict(class_weight="balanced", n_estimators=100, random_state=42)
    defaults.update(kwargs)
    return ExtraTreesClassifier(**defaults)


def create_voting_classifier(**kwargs) -> VotingClassifier:
    """Voting Classifier: soft voting with RF + ETC estimators."""
    rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=kwargs.pop("rf_n_estimators", 100),
        max_depth=kwargs.pop("rf_max_depth", None),
        random_state=42,
    )
    etc = ExtraTreesClassifier(
        class_weight="balanced",
        n_estimators=kwargs.pop("etc_n_estimators", 100),
        max_depth=kwargs.pop("etc_max_depth", None),
        random_state=42,
    )
    defaults = dict(voting="soft", estimators=[("rf", rf), ("etc", etc)])
    defaults.update(kwargs)
    return VotingClassifier(**defaults)


def create_gaussian_process(**kwargs) -> GaussianProcessClassifier:
    """Gaussian Process Classifier with RBF kernel."""
    defaults = dict(kernel=1.0 * RBF(1.0), random_state=42, max_iter_predict=200)
    defaults.update(kwargs)
    return GaussianProcessClassifier(**defaults)


def create_stacking_classifier(**kwargs) -> StackingClassifier:
    """Stacking Classifier: base [SVM, ETC, GPC], meta GPC."""
    svm = SVC(
        probability=True,
        class_weight="balanced",
        kernel=kwargs.pop("svm_kernel", "rbf"),
        C=kwargs.pop("svm_C", 1.0),
        random_state=42,
    )
    etc = ExtraTreesClassifier(
        class_weight="balanced",
        n_estimators=kwargs.pop("etc_n_estimators", 100),
        random_state=42,
    )
    gpc = GaussianProcessClassifier(
        kernel=1.0 * RBF(1.0), random_state=42, max_iter_predict=200
    )
    meta = GaussianProcessClassifier(
        kernel=1.0 * RBF(1.0), random_state=42, max_iter_predict=200
    )
    defaults = dict(
        estimators=[("svm", svm), ("etc", etc), ("gpc", gpc)],
        final_estimator=meta,
        cv=5,
    )
    defaults.update(kwargs)
    return StackingClassifier(**defaults)


# ---------------------------------------------------------------------------
# Bayesian hyperparameter search spaces (for scikit-optimize BayesSearchCV)
# ---------------------------------------------------------------------------


SEARCH_SPACES = {
    "lr": {
        "C": Real(1e-3, 100, prior="log-uniform"),
        "solver": Categorical(["lbfgs", "liblinear"]),
    },
    "etc": {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(3, 30),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
    },
    "vc_rf": {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(3, 30),
        "min_samples_split": Integer(2, 20),
    },
    "vc_etc": {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(3, 30),
        "min_samples_split": Integer(2, 20),
    },
    "gpc": {
        "kernel__k2__length_scale": Real(0.1, 10, prior="log-uniform"),
    },
    "sc_svm": {
        "C": Real(1e-2, 100, prior="log-uniform"),
        "kernel": Categorical(["rbf", "linear"]),
    },
    "sc_etc": {
        "n_estimators": Integer(50, 500),
        "max_depth": Integer(3, 30),
    },
}


# ---------------------------------------------------------------------------
# Registry of all models for easy iteration
# ---------------------------------------------------------------------------


def get_all_models() -> dict:
    """Return a dict of {name: (factory_fn, search_space_key)} for all 5 models."""
    return {
        "LR": (create_logistic_regression, "lr"),
        "ETC": (create_extra_trees, "etc"),
        "VC": (create_voting_classifier, None),  # tuned via sub-estimators
        "GPC": (create_gaussian_process, "gpc"),
        "SC": (create_stacking_classifier, None),  # tuned via sub-estimators
    }

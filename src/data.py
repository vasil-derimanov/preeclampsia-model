"""Data loading, preprocessing, synthetic data generation, and train-test split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    ALL_FEATURES,
    ASSISTED_REPRODUCTION_RATE,
    BINARY_PREVALENCES,
    CONCEPTION_METHOD_COLUMNS,
    CONCEPTION_METHOD_RAW,
    NON_PE_STATS,
    PE_STATS,
    PRETERM_PE_STATS,
    TARGETS,
)


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    return pd.read_csv(path)


def generate_synthetic_data(n_samples: int = 4644, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic data based on Table 1 statistics.

    Produces a dataset with roughly the same PE incidence (~4.5%) and
    preterm/term PE split as the original study.
    """
    rng = np.random.default_rng(seed)

    n_preterm_pe = max(1, round(n_samples * 49 / 4644))
    n_term_pe = max(1, round(n_samples * 161 / 4644))
    n_pe = n_preterm_pe + n_term_pe
    n_non_pe = n_samples - n_pe

    rows = []

    groups = [
        ("non_pe", n_non_pe, NON_PE_STATS, "non_pe"),
        ("preterm_pe", n_preterm_pe, PRETERM_PE_STATS, "preterm_pe"),
        ("term_pe", n_term_pe, PE_STATS, "pe"),  # term PE uses overall PE stats
    ]

    for group_label, n, stats, prev_key in groups:
        for _ in range(n):
            row = {}

            # Continuous features (clamp to physiologically plausible ranges)
            row["maternal_age"] = max(16, rng.normal(*stats["maternal_age"]))
            row["height"] = max(130, rng.normal(*stats["height"]))
            row["pre_pregnancy_weight"] = max(35, rng.normal(*stats["pre_pregnancy_weight"]))
            row["map_mmhg"] = max(50, rng.normal(*stats["map_mmhg"]))
            row["uta_pi"] = max(0.3, rng.normal(*stats["uta_pi"]))
            row["plgf"] = max(1, rng.normal(*stats["plgf"]))
            row["papp_a"] = max(0.1, rng.normal(*stats["papp_a"]))

            # Binary features
            for feat, (non_pe_rate, pe_rate, preterm_pe_rate) in BINARY_PREVALENCES.items():
                if prev_key == "non_pe":
                    rate = non_pe_rate
                elif prev_key == "preterm_pe":
                    rate = preterm_pe_rate
                else:
                    rate = pe_rate
                row[feat] = int(rng.random() < rate)

            # Conception method
            assisted_rate = ASSISTED_REPRODUCTION_RATE[prev_key]
            if rng.random() < assisted_rate:
                # Split assisted reproduction into ovulation induction (~30%) and IVF-ET (~70%)
                row[CONCEPTION_METHOD_RAW] = (
                    "ovulation_induction" if rng.random() < 0.3 else "ivf_et"
                )
            else:
                row[CONCEPTION_METHOD_RAW] = "natural"

            # Targets
            row["pe_all"] = int(group_label in ("preterm_pe", "term_pe"))
            row["preterm_pe"] = int(group_label == "preterm_pe")
            row["term_pe"] = int(group_label == "term_pe")

            rows.append(row)

    df = pd.DataFrame(rows)
    # Shuffle rows
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """One-hot encode conception_method and normalize features with MinMaxScaler.

    Returns the processed DataFrame and the fitted scaler.
    """
    df = df.copy()

    # One-hot encode conception_method if the raw column is present
    if CONCEPTION_METHOD_RAW in df.columns:
        dummies = pd.get_dummies(df[CONCEPTION_METHOD_RAW], prefix="conception")
        # Ensure all expected columns exist
        for col in CONCEPTION_METHOD_COLUMNS:
            if col not in dummies.columns:
                dummies[col] = 0
        df = pd.concat([df.drop(columns=[CONCEPTION_METHOD_RAW]), dummies[CONCEPTION_METHOD_COLUMNS]], axis=1)

    # Identify feature columns present in df
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]

    # MinMaxScaler normalization to [0, 1]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 stratified train-test split."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

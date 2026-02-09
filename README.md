# Preeclampsia Prediction Model

Machine learning pipeline for first-trimester preeclampsia (PE) prediction, based on Li et al. (2024), *Frontiers in Endocrinology*.

## Overview

Predicts preeclampsia risk from maternal characteristics, medical history, and biophysical/biochemical markers collected at 11-13+6 weeks' gestation. Implements five classifiers from the reference paper:

| Model | Description |
|-------|-------------|
| **LR** | Logistic Regression |
| **ETC** | Extra Trees Classifier |
| **VC** | Voting Classifier (RF + ETC, soft voting) |
| **GPC** | Gaussian Process Classifier (RBF kernel) |
| **SC** | Stacking Classifier (SVM + ETC + GPC base, GPC meta) |

## Project Structure

```
main.py                  # CLI entry point
requirements.txt         # Python dependencies
src/
    config.py            # Feature definitions, Table 1 statistics
    data.py              # Data loading, synthetic generation, preprocessing
    models.py            # 5 classifier factory functions + search spaces
    evaluation.py        # Metrics, ROC plots, calibration, SHAP
    pipeline.py          # Repeated CV, bootstrap CI, hyperparameter tuning
data/
    .gitkeep             # Placeholder for real dataset (CSV)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run with synthetic data (demo)

```bash
python main.py --synthetic
```

### Run with real data

```bash
python main.py --data data/your_dataset.csv
```

The CSV should contain columns matching the feature names in `src/config.py`, plus target columns (`pe_all`, `preterm_pe`, `term_pe`).

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--synthetic` | off | Use synthetic data generated from Table 1 statistics |
| `--data PATH` | none | Path to CSV dataset |
| `--n-samples N` | 4644 | Number of synthetic samples to generate |
| `--n-repeats N` | 5 | Number of CV repetitions (paper uses 200) |
| `--models M [M ...]` | all | Which models to run: `LR`, `ETC`, `VC`, `GPC`, `SC` |
| `--feature-sets F [F ...]` | `full` | Feature sets: `maternal_only`, `maternal_map`, `maternal_map_pappa`, `maternal_map_pappa_utapi`, `full` |
| `--targets T [T ...]` | `pe_all preterm_pe` | Targets: `pe_all`, `preterm_pe`, `term_pe` |
| `--output-dir DIR` | `output` | Directory for results |

### Examples

```bash
# Quick test with fast models only
python main.py --synthetic --n-repeats 2 --models LR ETC VC

# Full evaluation matching the paper (slow — GPC/SC take minutes per fold)
python main.py --synthetic --n-repeats 200

# Compare feature sets for Voting Classifier
python main.py --synthetic --models VC --feature-sets maternal_only maternal_map full
```

## Output

Results are saved to the `output/` directory:

- `{target}_{feature_set}_results.csv` — metrics with bootstrap 95% CIs (AUC, sensitivity, specificity, DR@10%FPR, DR@20%FPR, Brier score, calibration slope/intercept)
- `{target}_{feature_set}_roc.png` — ROC curves for all evaluated models

## Performance Notes

- **LR, ETC, VC**: Fast (seconds per CV run on 4644 samples)
- **GPC**: Slow (~500s for 10 folds on 4644 samples)
- **SC**: Slowest (uses GPC as both base and meta-learner)

For quick iteration, use `--models LR ETC VC` and a small `--n-repeats`.

## Reference

Li T, Xu M, Wang Y, et al. (2024). Prediction model of preeclampsia using machine learning based methods: a population based cohort study in China. *Front. Endocrinol.* 15:1345573. doi: 10.3389/fendo.2024.1345573

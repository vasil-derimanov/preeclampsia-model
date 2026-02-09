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

## Model Comparison

### How each model works

- **LR (Logistic Regression)** — Draws a linear boundary to separate PE from non-PE. Each feature gets a weight (e.g., "higher MAP increases PE risk by X"). Outputs probability directly from a formula.
- **ETC (Extra Trees Classifier)** — Ensemble of 100 random decision trees with randomized split points. Captures non-linear patterns and feature interactions automatically.
- **VC (Voting Classifier)** — Combines Random Forest + Extra Trees and averages their predicted probabilities (soft voting). Two ensembles making different mistakes cancel out each other's errors.
- **GPC (Gaussian Process Classifier)** — Probabilistic model that measures similarity between patients using an RBF kernel. Produces well-calibrated probabilities with uncertainty estimates. O(n^3) time complexity.
- **SC (Stacking Classifier)** — Two-level architecture. Level 1: SVM + ETC + GPC each make predictions. Level 2: a GPC meta-learner takes those predictions as inputs and makes the final decision.

### Results from the paper (4644 samples, full feature set)

| Metric | LR | ETC | VC | GPC | SC |
|--------|-----|-----|-----|-----|-----|
| **AUC (all PE)** | 0.824 | 0.817 | 0.831 | 0.832 | 0.825 |
| **AUC (preterm PE)** | 0.851 | 0.855 | **0.884** | 0.878 | 0.857 |
| **DR@10%FPR (preterm PE)** | 0.567 | 0.537 | **0.625** | 0.593 | 0.555 |
| **Speed (10 folds)** | **~0.1s** | ~1.3s | ~3.5s | ~500s | ~2000s+ |
| **Interpretability** | **high** | low | low | low | very low |
| **Non-linear patterns** | no | yes | yes | yes | yes |
| **Scalability** | excellent | good | good | poor (O(n^3)) | very poor |
| **Calibration** | good | moderate | moderate | **best** | moderate |

### Key takeaways

1. **VC is the best performer for preterm PE** — best AUC (0.884) and DR@10%FPR (0.625), catching 62.5% of preterm PE cases while only flagging 10% of healthy women.
2. **For all PE, models perform similarly** — AUCs cluster around 0.82-0.83; differences are within confidence intervals.
3. **GPC adds almost nothing over VC** despite being ~150x slower. Its one advantage is better probability calibration.
4. **SC underperforms expectations** — with only 49 preterm PE cases, the meta-learner overfits its own internal CV rather than learning which base model to trust.
5. **LR is the best value** — 95% of VC's performance at 0.3% of the compute time, and fully interpretable.

**Practical recommendation**: Use **VC** as the primary screening model and **LR** for clinical explainability. GPC and SC are not worth the compute cost for this dataset size.

For quick iteration, use `--models LR ETC VC` and a small `--n-repeats`.

## Reference

Li T, Xu M, Wang Y, et al. (2024). Prediction model of preeclampsia using machine learning based methods: a population based cohort study in China. *Front. Endocrinol.* 15:1345573. doi: 10.3389/fendo.2024.1345573

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
    template.csv         # CSV template to share with clinical team
    .gitkeep             # Placeholder for real dataset
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

A template CSV is provided at `data/template.csv`. Share it with your clinical team to fill in. The expected columns are:

**Maternal characteristics**

| Column | Type | Description |
|--------|------|-------------|
| `maternal_age` | number | Age in years (e.g., 29.5) |
| `height` | number | Height in cm (e.g., 162) |
| `pre_pregnancy_weight` | number | Weight in kg before pregnancy (e.g., 57) |
| `nulliparous` | 0 or 1 | First pregnancy = 1, not first = 0 |
| `conception_method` | text | `natural`, `ovulation_induction`, or `ivf_et` |
| `family_history_pe` | 0 or 1 | Family history of preeclampsia |
| `smoking` | 0 or 1 | Smoking status |

**Medical history**

| Column | Type | Description |
|--------|------|-------------|
| `history_pe` | 0 or 1 | Previous preeclampsia |
| `history_chronic_hypertension` | 0 or 1 | Chronic hypertension |
| `history_chronic_kidney_disease` | 0 or 1 | Chronic kidney disease |
| `history_diabetes` | 0 or 1 | Type 1 or type 2 diabetes |
| `history_sle_aps` | 0 or 1 | SLE or antiphospholipid syndrome |

**Biomarkers (measured at 11-13+6 weeks' gestation)**

| Column | Type | Description |
|--------|------|-------------|
| `map_mmhg` | number | Mean arterial pressure in mmHg (e.g., 82.6) |
| `uta_pi` | number | Uterine artery pulsatility index (e.g., 1.80) |
| `papp_a` | number | PAPP-A in IU/L (e.g., 4.56) |
| `plgf` | number | Placental growth factor in pg/mL (e.g., 33.8) |

**Outcomes**

| Column | Type | Description |
|--------|------|-------------|
| `pe_all` | 0 or 1 | Developed any preeclampsia |
| `preterm_pe` | 0 or 1 | Developed preterm PE (delivery <37 weeks) |
| `term_pe` | 0 or 1 | Developed term PE (delivery >=37 weeks) |

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

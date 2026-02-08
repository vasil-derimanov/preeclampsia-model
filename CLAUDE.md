# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a **preeclampsia (PE) prediction model** using machine learning, targeting first-trimester screening (11–13+6 weeks' gestation). The project is based on the methodology described in `fendo-15-1345573.pdf` (Li et al., 2024, *Frontiers in Endocrinology*).

**Status:** Early-stage — no application code has been written yet. The repo currently contains only the reference paper, license (MIT), and project scaffolding.

## Domain Context

Preeclampsia is a hypertensive pregnancy disorder affecting 2–5% of pregnancies worldwide. Early identification of high-risk patients enables aspirin intervention from 12 weeks' gestation. The prediction model combines:

- **Maternal characteristics:** age, height, pre-pregnancy weight, nulliparity, conception method, family history of PE, smoking status
- **Medical history:** prior PE, chronic hypertension, chronic kidney disease, diabetes (type 1/2), SLE/antiphospholipid syndrome
- **Biophysical markers:** mean arterial pressure (MAP), uterine artery pulsatility index (UtA-PI)
- **Biochemical markers:** PAPP-A, placental growth factor (PLGF)

PE is classified as **preterm PE** (delivery <37 weeks) and **term PE** (delivery ≥37 weeks).

## ML Algorithms from Reference Paper

The reference study evaluated five classifiers:
- Logistic Regression (LR)
- Extra Trees Classifier (ETC)
- Voting Classifier (VC) — best performance for preterm PE (AUC=0.884)
- Gaussian Process Classifier (GPC)
- Stacking Classifier (SC)

Validation approach: 5-fold cross-validation with 80/20 train-test split. Key metrics: AUC, detection rate at 10% false positive rate.

## Environment

- **Language:** Python 3.9
- **IDE:** PyCharm (JetBrains)
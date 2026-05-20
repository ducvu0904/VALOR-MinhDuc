# VALOR: Value-Aware Revenue Uplift Modeling

This repository contains the official implementation to reproduce the paper: **VALOR: Value-Aware Revenue Uplift Modeling with Treatment-Gated Representation for B2B Sales**, along with baseline frameworks like RERUM and Robust ZILN-GBDT.

## The VALOR Framework

Traditional uplift modeling often focuses on binary outcomes (e.g., conversion) or uses simple regression (MSE) for continuous outcomes. However, in scenarios like B2B sales or high-value e-commerce, the revenue distribution is **Zero-Inflated** (most customers don't buy) and **Heavy-Tailed** (a few customers buy huge amounts).

**VALOR** addresses these challenges through three core innovations:

1. **Focal-ZILN Loss (`--use_focal`)**: Adapts the Zero-Inflated Log-Normal (ZILN) loss by introducing a focal mechanism. It down-weights the massive number of "easy" non-converters, forcing the model to focus on hard-to-predict actual buyers and the long tail of revenue.
2. **Value-Weighted Ranking (`--use_ranking`)**: A pairwise ranking objective that directly optimizes the ranking of predicted uplift scores. It applies larger penalties when the model misranks pairs of customers who have a large actual revenue difference, aligning the model directly with financial impact.
3. **Treatment-Gated Interaction (GTI) (`--use_gating`)**: An architectural enhancement that allows the shared representation layers to adapt dynamically based on the assigned treatment, capturing complex non-linear interactions between covariates and the treatment variable.

## Decoupled Ranking Architecture

The architecture has been modularized so that complex ranking losses are decoupled from specific wrapper classes. 
- **Listwise Uplift Ranking (`--use_uplift_ranking`)** and **Pairwise Response Ranking (`--use_response_ranking`)** are natively supported on all standard DNN backbones (TARNet, DragonNet, CFR, etc.) without needing the RERUM wrapper.
- **RERUM Framework (`--rerum`)**: The full multi-objective ZILN + Pairwise + Listwise wrapper architecture is strictly triggered by this explicit flag.

## Environment Setup

The codebase is built in Python using PyTorch. Ensure you have the following dependencies installed:

```bash
pip install torch numpy pandas scipy scikit-learn econml matplotlib tqdm optuna
```

## Running Experiments (Optuna Tuning)

The primary entry point for reproducible evaluations is `scripts/tune.py`. This script performs rigorous hyperparameter tuning using Optuna (k-fold Monte Carlo on the validation set) and evaluates the best parameters on the held-out test set (N=10 seeds).

Results are automatically saved to `results/tuned/{dataset}.csv`.

### Key Command Line Arguments

*   `--model`: The base model architecture. Options: `TARNet`, `DragonNet`, `CFR-WASS`, `CFR-MMD`, `UniTE`, `EUEN`, `T-Learner`, `S-Learner`, `CausalForest`, `ZILN-GBDT`.
*   `--dataset`: Dataset to run. Options: `synthetic`, `hillstrom-men`, `hillstrom-women`.
*   `--use_ziln`: Enables the standard ZILN head (replaces standard MSE).
*   `--use_focal`: Upgrades the ZILN head to use **Focal-ZILN** loss.
*   `--use_gating`: Enables **Treatment-Gated Interaction (GTI)** on the backbone.
*   `--use_ranking`: Enables **Value-Weighted Ranking (WR)** loss.
*   `--use_uplift_ranking`: Enables listwise uplift ranking loss (available on all backbones).
*   `--use_response_ranking`: Enables pairwise response ranking loss (available on all backbones).
*   `--rerum`: Wraps the model in the **RERUM** architecture.
*   `--n_trials`: Number of Optuna trials for tuning (default: 50).

### Execution Examples

**1. Run Baseline Models:**
```bash
python scripts/tune.py --model TARNet --dataset hillstrom-men
python scripts/tune.py --model CFR-MMD --dataset hillstrom-women
```

**2. Run Standard Baselines with Decoupled Ranking:**
```bash
# TARNet with Listwise Uplift Ranking (natively on BaselineTrainer)
python scripts/tune.py --model TARNet --use_uplift_ranking

# DragonNet with Response Ranking
python scripts/tune.py --model DragonNet --use_response_ranking
```

**3. Run Partial VALOR Ablations:**
```bash
# TARNet with standard ZILN
python scripts/tune.py --model TARNet --use_ziln

# TARNet with Focal-ZILN and GTI
python scripts/tune.py --model TARNet --use_ziln --use_focal --use_gating
```

**4. Run Full VALOR Framework:**
```bash
python scripts/tune.py --model TARNet --use_ziln --use_focal --use_gating --use_ranking --dataset hillstrom-men
```

**5. Run RERUM Baseline:**
```bash
python scripts/tune.py --model TARNet --rerum --dataset hillstrom-women
```

## Running the Full Test Suite

To systematically run all major ablations and baseline comparisons across a specific dataset, use the overarching `tune_all.py` orchestration script:

```bash
# Run all ablations (VALOR variants, RERUM variants, Baselines) automatically
python scripts/tune_all.py --dataset hillstrom-men
```

The script manages a "skip-if-exists" mechanism to resume safely if interrupted. All aggregated metrics (AUUC, Qini, Lift@30, KRCC, Latency) will be accumulated in `results/tuned/<dataset>.csv`, tracking the mean and standard deviation across seeds.

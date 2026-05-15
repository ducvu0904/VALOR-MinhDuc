# VALOR: Value-Aware Revenue Uplift Modeling

This repository contains the official implementation to reproduce the paper: **VALOR: Value-Aware Revenue Uplift Modeling with Treatment-Gated Representation for B2B Sales**, along with baseline frameworks like RERUM and Robust ZILN-GBDT.

## The VALOR Framework

Traditional uplift modeling often focuses on binary outcomes (e.g., conversion) or uses simple regression (MSE) for continuous outcomes. However, in scenarios like B2B sales or high-value e-commerce, the revenue distribution is **Zero-Inflated** (most customers don't buy) and **Heavy-Tailed** (a few customers buy huge amounts).

**VALOR** addresses these challenges through three core innovations:

1. **Focal-ZILN Loss (`--use_focal`)**: Adapts the Zero-Inflated Log-Normal (ZILN) loss by introducing a focal mechanism. It down-weights the massive number of "easy" non-converters, forcing the model to focus on hard-to-predict actual buyers and the long tail of revenue.
2. **Value-Weighted Ranking (`--use_ranking`)**: A pairwise ranking objective that directly optimizes the ranking of predicted uplift scores. It applies larger penalties when the model misranks pairs of customers who have a large actual revenue difference, aligning the model directly with financial impact.
3. **Treatment-Gated Interaction (GTI) (`--use_gating`)**: An architectural enhancement that allows the shared representation layers to adapt dynamically based on the assigned treatment, capturing complex non-linear interactions between covariates and the treatment variable.

## Environment Setup

The codebase is built in Python using PyTorch. Ensure you have the following dependencies installed:

```bash
pip install torch numpy pandas scipy scikit-learn econml matplotlib tqdm
```

## Running the Experiments

The primary entry point is `main.py`. The synthetic dataset is generated automatically on the first run and uses fixed seeds to ensure exact reproducibility across multiple runs.

### Command Line Arguments

*   `--model`: The base model architecture. Options: `TARNet`, `DragonNet`, `CFR-WASS`, `CFR-MMD`, `UniTE`, `EUEN`, `T-Learner`, `S-Learner`, `CausalForest`, `ZILN-GBDT`.
*   `--use_ziln`: Enables the standard ZILN head (replaces standard MSE).
*   `--use_focal`: Upgrades the ZILN head to use **Focal-ZILN** loss.
*   `--use_gating`: Enables **Treatment-Gated Interaction (GTI)** on the backbone.
*   `--use_ranking`: Enables **Value-Weighted Ranking (WR)** loss.
*   `--rerum`: Runs the **RERUM** framework baseline (wraps the backbone in a multi-objective ZILN + Pairwise + Listwise architecture).

### Execution Examples

**1. Run Baseline Models:**
```bash
python3 main.py --model TARNet
python3 main.py --model CFR-MMD
```

**2. Run Partial VALOR Ablations:**
```bash
# TARNet with standard ZILN
python3 main.py --model TARNet --use_ziln

# TARNet with Focal-ZILN and GTI
python3 main.py --model TARNet --use_ziln --use_focal --use_gating
```

**3. Run Full VALOR Framework:**
```bash
python3 main.py --model TARNet --use_ziln --use_focal --use_gating --use_ranking
python3 main.py --model CFR-MMD --use_ziln --use_focal --use_gating --use_ranking
```

**4. Run RERUM Baseline:**
```bash
python3 main.py --model TARNet --rerum
```

**5. Run Tree-based Baselines:**
```bash
python3 main.py --model CausalForest
python3 main.py --model ZILN-GBDT
```

## Running the Full Test Suite

To run all major ablations and baseline comparisons automatically, you can use the following bash script:

```bash
#!/bin/bash

# 1. VALOR Variants (DNN + ZILN + Focal + GTI/WR)
models=("TARNet" "DragonNet" "CFR-MMD" "CFR-WASS")

for m in "${models[@]}"; do
    echo "Running $m + ZILN + Focal + GTI..."
    python3 main.py --model "$m" --use_ziln --use_focal --use_gating
    
    echo "Running $m + ZILN + Focal + WR..."
    python3 main.py --model "$m" --use_ziln --use_focal --use_ranking
    
    echo "Running Full VALOR ($m + ZILN + Focal + GTI + WR)..."
    python3 main.py --model "$m" --use_ziln --use_focal --use_gating --use_ranking
done

# 2. Specific Baseline
python3 main.py --model CFR-WASS --use_ziln --use_focal

# 3. RERUM Framework Baseline
rerum_models=("TARNet" "DragonNet" "CFR-MMD" "CFR-WASS")
for m in "${rerum_models[@]}"; do
    echo "Running $m + RERUM..."
    python3 main.py --model "$m" --rerum
done

# 4. Tree-based Models
echo "Running Robust ZILN-GBDT..."
python3 main.py --model ZILN-GBDT

echo "All experiments completed. Check results.csv for the aggregated metrics."
```

## Results

The aggregated metrics (AUUC, Qini, Lift@30, KRCC, Latency) are automatically appended to `results.csv` after each run. The current results were generated using the following experimental setup:

- **Data Scale**: 5,000 UIDs and 50,000 PIDs.
- **Interaction Context**: Each user is assigned 30 to 60 random product interactions (approx. 225k total samples).
- **Evaluation**: All metrics are reported as the mean and standard deviation across 5 fixed seeds (`[42, 123, 456, 789, 1024]`).

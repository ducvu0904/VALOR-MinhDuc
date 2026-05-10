#!/bin/bash

# 1. VALOR Variants (DNN + ZILN + Focal + GTI/WR)
models=("TARNet" "DragonNet" "CFR-MMD" "CFR-WASS")

for m in "${models[@]}"; do
    # ZILN + Focal + GTI
    python3 main.py --model "$m" --use_ziln --use_focal --use_gating
    # ZILN + Focal + WR
    python3 main.py --model "$m" --use_ziln --use_focal --use_ranking
    # Full VALOR (ZILN + Focal + GTI + WR)
    python3 main.py --model "$m" --use_ziln --use_focal --use_gating --use_ranking
done

# 2. Specific CFR-WASS Baseline
python3 main.py --model CFR-WASS --use_ziln --use_focal

# 3. RERUM Variants (Backbone + RERUM Wrapper)
rerum_models=("TARNet" "DragonNet" "CFR-MMD" "CFR-WASS")
for m in "${rerum_models[@]}"; do
    python3 main.py --model "$m" --rerum
done

# 4. Tree-based Model
python3 main.py --model ZILN-GBDT

echo "All experiments completed. Check results.csv for the full table."

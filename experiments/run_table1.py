"""
Reproduce Table 1 — Overall Performance (Synthetic Dataset).

Runs all models across 5 seeds and reports mean ± std for:
  AUUC, Qini, Lift@30, KRCC, Latency

Usage:
    python -m experiments.run_table1
"""

import sys
import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.generate import create_synthetic_data
from dataset.dataloader import get_dataloaders, get_cate_dims, _identify_columns

from models.baselines import (
    TARNet, DragonNet, CFR, UniTE, EUEN,
    TLearner, SLearner, CausalForestWrapper,
)
from models.valor_net import VALOR
from models.ziln_gbdt import ZILNGBDTForest

from training.trainer import BaselineTrainer, VALORTrainer
from training.evaluate import (
    evaluate_dnn_model, evaluate_tree_model, format_results_table,
)

# ------------------------------------------------------------------
#  Config
# ------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 30
LR = 5e-4
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Seeds: {SEEDS}")
print(f"Epochs: {EPOCHS}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
#  Generate data once
# ------------------------------------------------------------------
print("Generating synthetic data …")
df = create_synthetic_data()
print(f"Data shape: {df.shape}, sparsity: {(df['label']==0).mean():.2%}")


def run_dnn_model(model, train_loader, val_loader, test_loader, model_name,
                  use_focal=False, lambda_ipm=1.0):
    """Train & evaluate a single DNN model."""
    trainer = BaselineTrainer(
        model, lr=LR, epochs=EPOCHS, device=DEVICE,
        use_focal=use_focal, lambda_ipm=lambda_ipm,
    )
    trainer.train(train_loader, val_loader)
    metrics = evaluate_dnn_model(model, test_loader, device=DEVICE)
    return metrics


def run_valor_model(backbone, train_loader, val_loader, test_loader,
                    use_gating=True, use_ranking=True):
    """Train & evaluate a VALOR-wrapped model."""
    model = VALOR(backbone, use_gating=use_gating)
    trainer = VALORTrainer(
        model, lr=LR, epochs=EPOCHS, device=DEVICE,
        use_ranking=use_ranking,
    )
    trainer.train(train_loader, val_loader)
    metrics = evaluate_dnn_model(model, test_loader, device=DEVICE)
    return metrics


# ------------------------------------------------------------------
#  Model definitions
# ------------------------------------------------------------------

def build_models(cate_dims, num_count):
    """Return a dict of {name: (constructor_fn, is_valor, kwargs)}."""
    models = {}

    # === Meta-Learners ===
    models["T-Learner"] = lambda: TLearner(cate_dims, num_count)
    models["S-Learner"] = lambda: SLearner(cate_dims, num_count)

    # === Deep baselines (MSE) ===
    models["TARNet"] = lambda: TARNet(cate_dims, num_count)
    models["DragonNet"] = lambda: DragonNet(cate_dims, num_count)
    models["CFR-WASS"] = lambda: CFR(cate_dims, num_count, mode="wass")
    models["CFR-MMD"] = lambda: CFR(cate_dims, num_count, mode="mmd")
    models["UniTE"] = lambda: UniTE(cate_dims, num_count)
    models["EUEN"] = lambda: EUEN(cate_dims, num_count)

    # === RERUM variants (standard ZILN, no focal/ranking) ===
    models["RERUM-TARNet"] = lambda: TARNet(cate_dims, num_count, use_ziln=True)
    models["RERUM-DragonNet"] = lambda: DragonNet(cate_dims, num_count, use_ziln=True)
    models["RERUM-CFR-WASS"] = lambda: CFR(cate_dims, num_count, use_ziln=True, mode="wass")

    return models


def build_valor_models(cate_dims, num_count):
    """Return VALOR variant constructors."""
    models = {}

    models["VALOR-TARNet"] = lambda: TARNet(cate_dims, num_count, use_ziln=True)
    models["VALOR-DragonNet"] = lambda: DragonNet(cate_dims, num_count, use_ziln=True)
    models["VALOR-CFR-WASS"] = lambda: CFR(cate_dims, num_count, use_ziln=True, mode="wass")
    models["VALOR-CFR-MMD"] = lambda: CFR(cate_dims, num_count, use_ziln=True, mode="mmd")
    models["VALOR-UniTE"] = lambda: UniTE(cate_dims, num_count, use_ziln=True)
    models["VALOR-EUEN"] = lambda: EUEN(cate_dims, num_count, use_ziln=True)

    return models


# ------------------------------------------------------------------
#  Main loop
# ------------------------------------------------------------------

all_results = defaultdict(list)

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    set_seed(seed)
    train_loader, val_loader, test_loader, cate_dims, num_count = get_dataloaders(
        df, batch_size=BATCH_SIZE, seed=seed
    )

    # --- 1. Tree-based models ---
    # Prepare numpy data for tree models
    cat_cols, num_cols = _identify_columns(df)
    feature_cols = cat_cols + num_cols
    X_all = df[feature_cols].values
    t_all = df["treatment"].values
    y_all = df["label"].values

    N = len(df)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_test = int(N * 0.1)
    n_val = int(N * 0.2)
    n_train = N - n_test - n_val

    train_idx = perm[:n_train]
    test_idx = perm[n_train + n_val:]

    X_train, t_train, y_train = X_all[train_idx], t_all[train_idx], y_all[train_idx]
    X_test, t_test, y_test = X_all[test_idx], t_all[test_idx], y_all[test_idx]

    # Causal Forest
    print("Training Causal Forest …")
    cf = CausalForestWrapper(random_state=seed)
    cf.fit(X_train, t_train, y_train)
    metrics_cf = evaluate_tree_model(cf, X_test, t_test, y_test)
    all_results["Causal Forest"].append(metrics_cf)
    print(f"  Causal Forest: Qini={metrics_cf['qini']:.4f}")

    # ZILN-GBDT
    print("Training ZILN-GBDT …")
    gbdt = ZILNGBDTForest(random_state=seed)
    gbdt.fit(X_train, t_train, y_train)
    metrics_gbdt = evaluate_tree_model(gbdt, X_test, t_test, y_test)
    all_results["ZILN-GBDT"].append(metrics_gbdt)
    print(f"  ZILN-GBDT: Qini={metrics_gbdt['qini']:.4f}")

    # --- 2. DNN baselines ---
    baseline_models = build_models(cate_dims, num_count)
    for name, constructor in baseline_models.items():
        print(f"Training {name} …")
        set_seed(seed)
        model = constructor()
        is_rerum = name.startswith("RERUM")
        metrics = run_dnn_model(
            model, train_loader, val_loader, test_loader, name,
            use_focal=False,
        )
        all_results[name].append(metrics)
        print(f"  {name}: Qini={metrics['qini']:.4f}")

    # --- 3. VALOR variants ---
    valor_models = build_valor_models(cate_dims, num_count)
    for name, backbone_constructor in valor_models.items():
        print(f"Training {name} …")
        set_seed(seed)
        backbone = backbone_constructor()
        metrics = run_valor_model(
            backbone, train_loader, val_loader, test_loader,
            use_gating=True, use_ranking=True,
        )
        all_results[name].append(metrics)
        print(f"  {name}: Qini={metrics['qini']:.4f}")


# ------------------------------------------------------------------
#  Aggregate & print
# ------------------------------------------------------------------
print("\n\nAggregating results across seeds …\n")

final_results = {}
for name, metrics_list in all_results.items():
    agg = {}
    for key in metrics_list[0].keys():
        vals = [m[key] for m in metrics_list if m[key] is not None and not np.isnan(m[key])]
        if vals:
            agg[key] = np.mean(vals)
            agg[f"{key}_std"] = np.std(vals)
        else:
            agg[key] = np.nan
            agg[f"{key}_std"] = np.nan
    final_results[name] = agg

# Build table
rows = []
for name, m in final_results.items():
    rows.append({
        "Model": name,
        "AUUC": f"{m.get('auuc', 0):.4f} ± {m.get('auuc_std', 0):.4f}",
        "Qini": f"{m.get('qini', 0):.4f} ± {m.get('qini_std', 0):.4f}",
        "Lift@30": f"{m.get('lift_30', 0):.2f} ± {m.get('lift_30_std', 0):.2f}",
        "KRCC": f"{m.get('krcc', 0):.4f} ± {m.get('krcc_std', 0):.4f}",
        "Latency (ms)": f"{m.get('latency_ms', 0):.4f}",
    })

result_df = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("  TABLE 1: Overall Performance (Synthetic Dataset)")
print("=" * 80)
print(result_df.to_string(index=False))
print("=" * 80)

# Save to CSV
result_df.to_csv("experiments/table1_results.csv", index=False)
print("\nResults saved to experiments/table1_results.csv")

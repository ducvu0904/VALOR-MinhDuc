"""
Reproduce Table 2 — Ablation Study (Synthetic Dataset).

Progressively toggles VALOR components on each backbone:
  1. Baseline (MSE)
  2. +ZILN       (standard ZILN loss)
  3. +ZILN+Focal (Focal-ZILN loss)
  4. +ZILN+Focal+GTI (add Treatment-Gated Interaction)
  5. +ZILN+Focal+GTI+WR (full VALOR = add Value-Weighted Ranking)

Usage:
    python -m experiments.run_table2
"""

import sys
import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.generate import create_synthetic_data
from dataset.dataloader import get_dataloaders

from models.baselines import TARNet, DragonNet, CFR, UniTE, EUEN
from models.valor_net import VALOR

from training.trainer import BaselineTrainer, VALORTrainer
from training.evaluate import evaluate_dnn_model

# ------------------------------------------------------------------
#  Config
# ------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 30
LR = 5e-4
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
#  Data
# ------------------------------------------------------------------
print("Generating synthetic data …")
df = create_synthetic_data()
print(f"Data shape: {df.shape}")


# ------------------------------------------------------------------
#  Backbone definitions
# ------------------------------------------------------------------

BACKBONES = {
    "TARNet": TARNet,
    "DragonNet": DragonNet,
    "CFR-WASS": lambda cd, nc, **kw: CFR(cd, nc, mode="wass", **kw),
    "CFR-MMD": lambda cd, nc, **kw: CFR(cd, nc, mode="mmd", **kw),
    "UniTE": UniTE,
    "EUEN": EUEN,
}

ABLATION_CONFIGS = [
    # (label, use_ziln, use_focal, use_gating, use_ranking)
    ("Baseline (MSE)",           False, False, False, False),
    ("+ZILN",                    True,  False, False, False),
    ("+ZILN+Focal",              True,  True,  False, False),
    ("+ZILN+Focal+GTI",          True,  True,  True,  False),
    ("+ZILN+Focal+GTI+WR (VALOR)", True,  True,  True,  True),
]


# ------------------------------------------------------------------
#  Run
# ------------------------------------------------------------------
all_results = defaultdict(lambda: defaultdict(list))

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    set_seed(seed)
    train_loader, val_loader, test_loader, cate_dims, num_count = get_dataloaders(
        df, batch_size=BATCH_SIZE, seed=seed
    )

    for backbone_name, BackboneClass in BACKBONES.items():
        for config_label, use_ziln, use_focal, use_gating, use_ranking in ABLATION_CONFIGS:
            run_name = f"{backbone_name} | {config_label}"
            print(f"  Training: {run_name}")
            set_seed(seed)

            # Build backbone
            if callable(BackboneClass) and backbone_name.startswith("CFR"):
                backbone = BackboneClass(cate_dims, num_count, use_ziln=use_ziln)
            else:
                backbone = BackboneClass(cate_dims, num_count, use_ziln=use_ziln)

            if use_gating or use_ranking:
                # VALOR wrapper needed
                if not use_ziln:
                    # GTI/WR require ZILN; skip this impossible config
                    continue
                model = VALOR(backbone, use_gating=use_gating)
                trainer = VALORTrainer(
                    model, lr=LR, epochs=EPOCHS, device=DEVICE,
                    use_ranking=use_ranking,
                )
                trainer.train(train_loader, val_loader)
                metrics = evaluate_dnn_model(model, test_loader, device=DEVICE)
            else:
                # Pure baseline (with or without ZILN/Focal)
                trainer = BaselineTrainer(
                    backbone, lr=LR, epochs=EPOCHS, device=DEVICE,
                    use_focal=use_focal,
                )
                trainer.train(train_loader, val_loader)
                metrics = evaluate_dnn_model(backbone, test_loader, device=DEVICE)

            all_results[backbone_name][config_label].append(metrics)
            print(f"    AUUC={metrics['auuc']:.4f}, Qini={metrics['qini']:.4f}")


# ------------------------------------------------------------------
#  Aggregate & print
# ------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("  TABLE 2: Ablation Study (Synthetic Dataset)")
print("=" * 80)

rows = []
for backbone_name in BACKBONES.keys():
    for config_label, _, _, _, _ in ABLATION_CONFIGS:
        metrics_list = all_results[backbone_name].get(config_label, [])
        if not metrics_list:
            continue

        agg = {}
        for key in metrics_list[0].keys():
            vals = [m[key] for m in metrics_list if m[key] is not None and not np.isnan(m[key])]
            agg[key] = np.mean(vals) if vals else np.nan
            agg[f"{key}_std"] = np.std(vals) if vals else np.nan

        rows.append({
            "Backbone": backbone_name,
            "Config": config_label,
            "AUUC": f"{agg.get('auuc', 0):.4f} ± {agg.get('auuc_std', 0):.4f}",
            "Qini": f"{agg.get('qini', 0):.4f} ± {agg.get('qini_std', 0):.4f}",
            "Lift@30": f"{agg.get('lift_30', 0):.2f} ± {agg.get('lift_30_std', 0):.2f}",
            "KRCC": f"{agg.get('krcc', 0):.4f} ± {agg.get('krcc_std', 0):.4f}",
        })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))
print("=" * 80)

result_df.to_csv("experiments/table2_results.csv", index=False)
print("\nResults saved to experiments/table2_results.csv")

"""
run_all.py — Untuned baseline sweep.

Runs all backbone × ablation combinations using default hyperparameters
from config.default_hparams().  Results are saved to results/untuned/.

Usage:
    python scripts/run_all.py                          # all datasets
    python scripts/run_all.py --dataset hillstrom-men  # single dataset
"""

import subprocess
import argparse
import os

def run_cmd(cmd_list):
    print(f"\n>>> Executing: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all backbones on a specific dataset (untuned baselines)."
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "synthetic", "hillstrom-men", "hillstrom-women"],
        help="Which dataset to run (default: all)"
    )
    parser.add_argument(
        "--checkpoint_metric", "--checkpoint_restore", type=str, default="val_loss",
        dest="checkpoint_metric",
        choices=["val_loss", "val_qini"],
        help="Metric to select best epoch checkpoint during training (default: val_loss)."
    )
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["hillstrom-men", "hillstrom-women"]
    else:
        datasets = [args.dataset]

    # Hardcoded backbones
    backbones = ["TARNet", "DragonNet", "CFR-MMD", "CFR-WASS"]

    print(f"Targeting Datasets: {datasets}")
    print(f"Targeting Backbones: {backbones}")
    print(f"Checkpoint Metric: {args.checkpoint_metric}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(script_dir, "main.py")

    for d in datasets:
        for b in backbones:
            print(f"\n{'='*60}\n  Processing Backbone: {b} on Dataset: {d}\n{'='*60}")

            base_cmd = ["python", main_py_path, "--model", b, "--dataset", d,
                        "--checkpoint_metric", args.checkpoint_metric]

            # ── VALOR / Baseline ablations ─────────────────────────────────
            # 0. Backbone (MSE)
            run_cmd(base_cmd)
            # 1. Backbone + ZILN
            run_cmd(base_cmd + ["--use_ziln"])
            # 2. Backbone + ZILN + Focal
            run_cmd(base_cmd + ["--use_ziln", "--use_focal"])
            # 3. Backbone + ZILN + Focal + WR
            run_cmd(base_cmd + ["--use_ziln", "--use_focal", "--use_ranking"])
            # 4. Backbone + ZILN + Focal + GTI
            run_cmd(base_cmd + ["--use_ziln", "--use_focal", "--use_gating"])
            # 5. Full VALOR: Backbone + ZILN + Focal + GTI + WR
            run_cmd(base_cmd + ["--use_ziln", "--use_focal", "--use_gating", "--use_ranking"])

            # ── Ranking-only ablations (non-RERUM) ────────────────────────
            # 6. Backbone (MSE) + uplift ranking
            run_cmd(base_cmd + ["--use_uplift_ranking"])
            # 7. Backbone (MSE) + uplift ranking + response ranking
            run_cmd(base_cmd + ["--use_uplift_ranking", "--use_response_ranking"])
            # 8. Backbone + ZILN + uplift ranking
            run_cmd(base_cmd + ["--use_ziln", "--use_uplift_ranking"])

            # ── RERUM: requires ZILN + both ranking losses ─────────────────
            # 9. Full RERUM
            run_cmd(base_cmd + ["--rerum", "--use_ziln",
                                 "--use_uplift_ranking", "--use_response_ranking"])

    # ── EFIN experiments (self-contained, no backbone flags) ──────────────────
    for d in datasets:
        print(f"\n{'='*60}\n  Processing EFIN on Dataset: {d}\n{'='*60}")
        efin_base = ["python", main_py_path, "--model", "EFIN", "--dataset", d,
                     "--checkpoint_metric", args.checkpoint_metric]
        # 1. EFIN (MSE)
        run_cmd(efin_base)
        # 2. EFIN + ZILN
        run_cmd(efin_base + ["--use_ziln"])
        # 3. EFIN + Focal-ZILN
        run_cmd(efin_base + ["--use_ziln", "--use_focal"])


if __name__ == "__main__":
    main()


"""
tune_all_max_auqc.py — Sweeping Optuna tuning across models and datasets maximizing validation AUQC.

Runs all backbone × ablation combinations using scripts/tune_max_auqc.py.
Results are saved to results/tuned_max_auqc/.

Usage:
    python scripts/tune_all_max_auqc.py                          # all datasets, default 50 trials
    python scripts/tune_all_max_auqc.py --dataset hillstrom-men  # single dataset
    python scripts/tune_all_max_auqc.py --n_trials 20            # change number of trials
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
        description="Run Optuna tuning (maximizing AUQC) for all backbones on a specific dataset."
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "synthetic", "hillstrom-men", "hillstrom-women"],
        help="Which dataset to run (default: all)"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Number of Optuna trials per model (default: 50)"
    )
    parser.add_argument(
        "--checkpoint_metric", type=str, default="val_qini",
        choices=["val_loss", "val_qini"],
        help="Metric to select best epoch checkpoint during training."
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
    print(f"Trials per tuning run: {args.n_trials}")
    print(f"Checkpoint metric: {args.checkpoint_metric}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tune_py_path = os.path.join(script_dir, "tune_max_auqc.py")

    for d in datasets:
        for b in backbones:
            print(f"\n{'='*60}\n  Tuning Backbone: {b} on Dataset: {d} (Max AUQC)\n{'='*60}")

            base_cmd = ["python", tune_py_path, "--model", b, "--dataset", d,
                        "--n_trials", str(args.n_trials),
                        "--checkpoint_metric", args.checkpoint_metric]

            # ── VALOR / Baseline ablations ─────────────────────────────────
            # 1. Backbone (MSE)
            run_cmd(base_cmd)
            # 2. Backbone + ZILN
            run_cmd(base_cmd + ["--use_ziln"])
            # 3. Backbone + ZILN + Focal
            run_cmd(base_cmd + ["--use_ziln", "--use_focal"])
            # 4. Backbone + ZILN + Focal + WR
            run_cmd(base_cmd + ["--use_ziln", "--use_focal", "--use_ranking"])
            # 5. Backbone + ZILN + Focal + GTI
            run_cmd(base_cmd + ["--use_ziln", "--use_focal", "--use_gating"])
            # 6. Full VALOR: Backbone + ZILN + Focal + GTI + WR
            run_cmd(base_cmd + ["--use_ziln", "--use_focal", "--use_gating", "--use_ranking"])

            # ── Ranking-only ablations (non-RERUM) ────────────────────────
            # 7. Backbone (MSE) + uplift ranking
            run_cmd(base_cmd + ["--use_uplift_ranking"])
            # 8. Backbone (MSE) + uplift ranking + response ranking
            run_cmd(base_cmd + ["--use_uplift_ranking", "--use_response_ranking"])
            # 9. Backbone + ZILN + uplift ranking
            run_cmd(base_cmd + ["--use_ziln", "--use_uplift_ranking"])

            # ── RERUM: requires ZILN + both ranking losses ─────────────────
            # 10. Full RERUM
            run_cmd(base_cmd + ["--rerum", "--use_ziln",
                                 "--use_uplift_ranking", "--use_response_ranking"])

    # ── EFIN experiments (self-contained, no backbone flags) ──────────────────
    for d in datasets:
        print(f"\n{'='*60}\n  Tuning EFIN on Dataset: {d} (Max AUQC)\n{'='*60}")
        efin_base = ["python", tune_py_path, "--model", "EFIN", "--dataset", d,
                     "--n_trials", str(args.n_trials),
                     "--checkpoint_metric", args.checkpoint_metric]
        # 1. EFIN (MSE)
        run_cmd(efin_base)
        # 2. EFIN + ZILN
        run_cmd(efin_base + ["--use_ziln"])
        # 3. EFIN + Focal-ZILN
        run_cmd(efin_base + ["--use_ziln", "--use_focal"])


if __name__ == "__main__":
    main()

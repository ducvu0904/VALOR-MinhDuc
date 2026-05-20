"""
tune_all.py — Sweeping Optuna tuning across models and datasets.

Runs all backbone × ablation combinations using scripts/tune.py.
Results are saved to results/tuned/.

Usage:
    python scripts/tune_all.py                          # all datasets, default 50 trials
    python scripts/tune_all.py --dataset hillstrom-men  # single dataset
    python scripts/tune_all.py --n_trials 20            # change number of trials
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
        description="Run Optuna tuning for all backbones on a specific dataset."
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tune_py_path = os.path.join(script_dir, "tune.py")

    for d in datasets:
        for b in backbones:
            print(f"\n{'='*60}\n  Tuning Backbone: {b} on Dataset: {d}\n{'='*60}")

            base_cmd = ["python", tune_py_path, "--model", b, "--dataset", d,
                        "--n_trials", str(args.n_trials)]

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
            # 8. Backbone + ZILN + uplift ranking
            run_cmd(base_cmd + ["--use_ziln", "--use_uplift_ranking"])
            # 9. Backbone (MSE) + response ranking
            run_cmd(base_cmd + ["--use_response_ranking"])
            # 10. Backbone + ZILN + response ranking
            run_cmd(base_cmd + ["--use_ziln", "--use_response_ranking"])

            # ── RERUM: requires ZILN + both ranking losses ─────────────────
            # 11. Full RERUM
            run_cmd(base_cmd + ["--rerum", "--use_ziln",
                                 "--use_uplift_ranking", "--use_response_ranking"])


if __name__ == "__main__":
    main()
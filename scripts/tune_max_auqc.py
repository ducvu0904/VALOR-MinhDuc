import sys
import os
# Ensure the root project directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
tune_max_auqc.py — Optuna hyperparameter search for VALOR / RERUM models maximizing validation AUQC.

Usage:
    # Tune EFIN on hillstrom-men:
    python scripts/tune_max_auqc.py --model EFIN --dataset hillstrom-men \
                   --use_ziln --use_focal \
                   --n_trials 50

Results are saved to:
    results/tuned_max_auqc/{dataset_tag}.csv                    — final 10-seed metrics
    results/tuned_max_auqc/best_params/{model}_{dataset}.json   — best hparams
"""

import argparse
import json
import warnings
from collections import defaultdict

import numpy as np
# pyrefly: ignore [missing-import]
import optuna
# pyrefly: ignore [missing-import]
from optuna.samplers import TPESampler

from config import Config, default_hparams
from scripts.main import (
    get_model_name,
    load_data,
    run_single_experiment,
    get_project_root
)

# Suppress optuna's verbose per-trial logging; we print our own summary
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Seeds ─────────────────────────────────────────────────────────────────────
INNER_SEEDS = [42, 123, 456]       # k=3: used inside each Optuna trial
OUTER_SEEDS = list(range(10))      # N=10: used for the final evaluation


# ── Optuna search space ───────────────────────────────────────────────────────

def build_trial_hparams(trial, args) -> dict:
    """
    Sample a hyperparameter configuration from an Optuna trial.
    """
    hparams = default_hparams()  # start from defaults; override with trial values

    # ── Universal params ──────────────────────────────────────────────────────
    hparams["lr"]         = trial.suggest_float("lr",      1e-5, 1e-2, log=True)
    hparams["l2_reg"]     = trial.suggest_float("l2_reg",  1e-6, 1e-2, log=True)
    hparams["hidden_dim"] = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])

    # ── VALOR focal params (only when focal loss is in use) ───────────────────
    is_valor_focal = args.use_focal or args.use_gating or args.use_ranking
    if is_valor_focal:
        hparams["focal_gamma"] = trial.suggest_float("focal_gamma", 0.0, 5.0, step=0.5)
        hparams["focal_alpha"] = trial.suggest_float("focal_alpha", 0.5, 0.99)

    # ── Ranking weight (VALOR WR or RERUM response ranking) ───────────────────
    if args.use_ranking or args.use_response_ranking:
        hparams["lambda_rank"] = trial.suggest_float("lambda_rank", 1e-6, 1e-1, log=True)

    # ── Listwise uplift ranking weight ────────────────────────────────────────
    if getattr(args, "use_uplift_ranking", False):
        hparams["lambda_lu"] = trial.suggest_float("lambda_lu", 1e-6, 1e-1, log=True)

    # ── EFIN intervention constraint weight ───────────────────────────────────
    if args.model == "EFIN":
        hparams["lambda_c"] = trial.suggest_float("lambda_c", 0.01, 10.0, log=True)

    return hparams


# ── Optuna objective ──────────────────────────────────────────────────────────

def _make_loaders(args, hparams: dict, data: dict):
    """Build DataLoaders with the given batch_size. Returns (tl, vl, tel, cate_dims, num_count)."""
    from dataset.dataloader import get_dataloaders, get_dataloaders_from_splits
    batch_size = hparams["batch_size"]
    if data["is_split_dataset"]:
        return get_dataloaders_from_splits(
            data["train_df"], data["val_df"], data["test_df"],
            batch_size=batch_size,
        )
    else:
        return get_dataloaders(data["df"], batch_size=batch_size, seed=42)


def objective(trial, args, data: dict) -> float:
    """
    Optuna objective function.

    Performs an internal Monte Carlo evaluation over INNER_SEEDS (k=3)
    and returns the mean AUQC (qini) on the VALIDATION SET.
    Evaluating on val (not test) avoids test-set leakage during search.
    """
    hparams = build_trial_hparams(trial, args)

    is_valor_focal = args.use_focal or args.use_gating or args.use_ranking
    print(
        f"\n[Trial {trial.number}]  "
        f"lr={hparams['lr']:.2e}  l2={hparams['l2_reg']:.2e}  "
        f"bs={hparams['batch_size']}  hidden={hparams['hidden_dim']}"
        + (f"  gamma={hparams['focal_gamma']:.1f}  alpha={hparams['focal_alpha']:.2f}"
           if is_valor_focal else "")
    )

    is_tree = args.model in ["CausalForest", "ZILN-GBDT"]

    # Re-build dataloaders with trial's batch_size (tree models don't use them)
    if not is_tree:
        tl, vl, tel, cate_dims, num_count = _make_loaders(args, hparams, data)
    else:
        tl = vl = tel = cate_dims = num_count = None

    auqc_scores = []
    val_losses = []
    for seed in INNER_SEEDS:
        try:
            if is_tree:
                metrics = run_single_experiment(
                    seed, args, hparams,
                    X_train=data["X_train"], t_train=data["t_train"], y_train=data["y_train"],
                    X_test=data["X_test"],   t_test=data["t_test"],   y_test=data["y_test"],
                )
            else:
                metrics = run_single_experiment(
                    seed, args, hparams,
                    train_loader=tl, val_loader=vl, test_loader=tel,
                    cate_dims=cate_dims, num_count=num_count,
                    eval_loader=vl,   # ← evaluate on VAL SET during tuning
                )
            auqc_scores.append(metrics["qini"])
            val_losses.append(metrics.get("val_loss", 0.0))
        except Exception as e:
            warnings.warn(f"Trial {trial.number} seed {seed} failed: {e}")
            return float("-inf")

    mean_auqc = float(np.mean(auqc_scores))
    mean_val_loss = float(np.mean(val_losses))
    print(f"[Trial {trial.number}] Val Loss: {mean_val_loss:.4f} | Val AUQC: {mean_auqc:.4f}")
    return mean_auqc


# ── Final evaluation after Optuna ─────────────────────────────────────────────

def final_evaluation(args, best_hparams: dict, data: dict) -> dict:
    """
    Re-evaluate the best hparams with OUTER_SEEDS (N=10) on the TEST SET.

    Hyperparameters were selected using val AUQC; this is the held-out
    test evaluation that produces the final reported numbers.

    Returns aggregated metrics dict: {metric: [values over 10 seeds]}.
    """
    is_tree = args.model in ["CausalForest", "ZILN-GBDT"]

    if not is_tree:
        tl, vl, tel, cate_dims, num_count = _make_loaders(args, best_hparams, data)
    else:
        tl = vl = tel = cate_dims = num_count = None

    all_metrics = defaultdict(list)

    for seed in OUTER_SEEDS:
        print(f"\n=== Final Evaluation Seed {seed} (test set) ===")
        if is_tree:
            metrics = run_single_experiment(
                seed, args, best_hparams,
                X_train=data["X_train"], t_train=data["t_train"], y_train=data["y_train"],
                X_test=data["X_test"],   t_test=data["t_test"],   y_test=data["y_test"],
            )
        else:
            metrics = run_single_experiment(
                seed, args, best_hparams,
                train_loader=tl, val_loader=vl, test_loader=tel,
                cate_dims=cate_dims, num_count=num_count,
                eval_loader=tel,  # ← evaluate on TEST SET for final report
            )
        for k, v in metrics.items():
            if v is not None:
                all_metrics[k].append(v)

    return all_metrics


# ── Results helpers ───────────────────────────────────────────────────────────

def _get_tuned_output_path(args) -> str:
    mapping = {
        "synthetic":      "results/tuned_max_auqc/synthesis.csv",
        "hillstrom-men":  "results/tuned_max_auqc/hillstrom_men.csv",
        "hillstrom-women": "results/tuned_max_auqc/hillstrom_women.csv",
    }
    return mapping.get(args.dataset, f"results/tuned_max_auqc/{args.dataset}.csv")


def _get_best_params_path(args) -> str:
    model_name = get_model_name(args)
    safe_model   = model_name.replace(" ", "_").replace("/", "_").replace("+", "plus")
    safe_dataset = args.dataset.replace("-", "_")
    return f"results/tuned_max_auqc/best_params/{safe_model}_{safe_dataset}.json"


def save_tuned_csv(model_name: str, all_metrics: dict, filename: str):
    """Save tuned results with mean and std columns per metric."""
    import pandas as pd
    row = {"Model": model_name}
    for k, v_list in all_metrics.items():
        row[f"{k}_mean"] = float(np.mean(v_list))
        row[f"{k}_std"]  = float(np.std(v_list))

    df_new = pd.DataFrame([row])
    
    root_dir = get_project_root()
    full_filename = os.path.join(root_dir, filename)
    os.makedirs(os.path.dirname(full_filename), exist_ok=True)

    if not os.path.exists(full_filename):
        df_new.to_csv(full_filename, index=False)
    else:
        df_new.to_csv(full_filename, mode="a", header=False, index=False)
    print(f"Tuned results appended to {full_filename}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for VALOR / RERUM models maximizing validation AUQC."
    )
    # Model / dataset flags — mirrors main.py
    parser.add_argument("--model", type=str, required=True,
                        choices=["TARNet", "DragonNet", "CFR-WASS", "CFR-MMD",
                                 "UniTE", "EUEN", "T-Learner", "S-Learner",
                                 "CausalForest", "ZILN-GBDT", "EFIN"])
    parser.add_argument("--dataset", type=str, default="hillstrom-men",
                        choices=["synthetic", "hillstrom-men", "hillstrom-women"])
    parser.add_argument("--rerum",                action="store_true")
    parser.add_argument("--use_ziln",             action="store_true")
    parser.add_argument("--use_focal",            action="store_true")
    parser.add_argument("--use_gating",           action="store_true")
    parser.add_argument("--use_ranking",          action="store_true")
    parser.add_argument("--use_uplift_ranking",   action="store_true")
    parser.add_argument("--use_response_ranking", action="store_true")
    parser.add_argument("--checkpoint_metric", "--checkpoint_restore", type=str, default="val_qini",
                        dest="checkpoint_metric",
                        choices=["val_loss", "val_qini"],
                        help="Metric to select best epoch checkpoint during training.")

    # Tuning-specific flags
    parser.add_argument("--n_trials",   type=int, default=50,
                        help="Number of Optuna trials (default: 50).")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optional Optuna study name for resuming.")
    parser.add_argument("--storage",    type=str, default=None,
                        help="Optional Optuna DB URL, e.g. sqlite:///optuna.db")

    args = parser.parse_args()

    # Auto-enable ziln when required by other flags
    if (args.use_focal or args.use_gating or args.use_ranking) and not args.use_ziln:
        print("Warning: use_focal/use_gating/use_ranking require use_ziln. Enabling it.")
        args.use_ziln = True

    model_name = get_model_name(args)
    
    # Early Exit: If this model's best_params JSON already exists, skip it entirely!
    best_params_path = _get_best_params_path(args)
    root_dir = get_project_root()
    full_best_params_path = os.path.join(root_dir, best_params_path)
    
    if os.path.exists(full_best_params_path):
        print(f"\n[SKIP] Model {model_name} on {args.dataset} already tuned! Skipping...")
        return

    print(f"\n{'='*60}")
    print(f"  Optuna Tuning (Max AUQC): {model_name}")
    print(f"  Dataset                 : {args.dataset}")
    print(f"  Trials                  : {args.n_trials}")
    print(f"  Inner seeds             : {INNER_SEEDS}  (k={len(INNER_SEEDS)})")
    print(f"  Outer seeds             : N={len(OUTER_SEEDS)}")
    print(f"{'='*60}\n")

    # ── 1. Load data once (shared across all trials) ──────────────────────────
    data = load_data(args, default_hparams())

    # ── 2. Create / resume Optuna study ──────────────────────────────────────
    study_name = args.study_name or f"{model_name}_{args.dataset}_max_auqc".replace(" ", "_")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=study_name,
        storage=args.storage,
        load_if_exists=(args.storage is not None),
    )

    study.optimize(
        lambda trial: objective(trial, args, data),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ── 3. Report best trial ──────────────────────────────────────────────────
    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best Trial #{best_trial.number}  |  Mean Val AUQC: {best_trial.value:.4f}")
    print(f"  Best Params:")
    for k, v in best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # Merge Optuna best params on top of full defaults
    best_hparams = default_hparams()
    best_hparams.update(best_trial.params)

    # ── 4. Save best params JSON ──────────────────────────────────────────────
    best_params_path = _get_best_params_path(args)
    root_dir = get_project_root()
    full_best_params_path = os.path.join(root_dir, best_params_path)
    os.makedirs(os.path.dirname(full_best_params_path), exist_ok=True)
    with open(full_best_params_path, "w") as f:
        json.dump(best_hparams, f, indent=2)
    print(f"Best hyperparameters saved to {full_best_params_path}")

    # ── 5. Final evaluation: N=10 seeds ──────────────────────────────────────
    print(f"\nRunning final evaluation with N={len(OUTER_SEEDS)} seeds...")
    final_metrics = final_evaluation(args, best_hparams, data)

    # ── 6. Print final aggregated results ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL TUNED RESULTS (Max AUQC): {model_name}")
    print(f"{'='*60}")
    for k, v_list in final_metrics.items():
        mean_v = np.mean(v_list)
        std_v  = np.std(v_list)
        unit   = " ms" if k == "latency_ms" else ""
        print(f"  {k}: {mean_v:.4f} ± {std_v:.4f}{unit}")
    print(f"{'='*60}\n")

    # ── 7. Save tuned CSV ────────────────────────────────────────────────────
    out_path = _get_tuned_output_path(args)
    save_tuned_csv(model_name, final_metrics, filename=out_path)


if __name__ == "__main__":
    main()

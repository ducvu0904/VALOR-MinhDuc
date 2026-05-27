import sys
import os
# Ensure the root project directory is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

from dataset.synthetic.generate import create_synthetic_data
from dataset.dataloader import get_dataloaders, get_dataloaders_from_splits, _identify_columns

from models.baselines import TARNet, DragonNet, CFR, UniTE, EUEN, TLearner, SLearner, CausalForestWrapper
from models.valor.valor_net import VALOR
from models.rerum.rerum_net import RERUMWrapper
from models.ziln_gbdt import ZILNGBDTForest
from models.efin import EFIN

from training.trainer import BaselineTrainer, VALORTrainer, RERUMTrainer, EFINTrainer
from training.evaluate import evaluate_dnn_model, evaluate_tree_model
from config import Config, default_hparams


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
#  Model / Trainer factory helpers
# =====================================================================

def _build_dnn_backbone(args, cate_dims, num_count, hparams: dict, use_ziln: bool = None):
    """Build a DNN backbone from args + hparams."""
    hidden = hparams["hidden_dim"]
    if use_ziln is None:
        use_ziln = args.use_ziln

    if args.model == "TARNet":
        return TARNet(cate_dims, num_count, hidden=hidden, use_ziln=use_ziln)
    elif args.model == "DragonNet":
        return DragonNet(cate_dims, num_count, hidden=hidden, use_ziln=use_ziln)
    elif args.model == "CFR-WASS":
        return CFR(cate_dims, num_count, hidden=hidden, mode="wass", use_ziln=use_ziln)
    elif args.model == "CFR-MMD":
        return CFR(cate_dims, num_count, hidden=hidden, mode="mmd", use_ziln=use_ziln)
    elif args.model == "UniTE":
        return UniTE(cate_dims, num_count, hidden=hidden, use_ziln=use_ziln)
    elif args.model == "EUEN":
        return EUEN(cate_dims, num_count, hidden=hidden, use_ziln=use_ziln)
    elif args.model == "T-Learner":
        return TLearner(cate_dims, num_count, hidden=hidden, use_ziln=use_ziln)
    elif args.model == "S-Learner":
        return SLearner(cate_dims, num_count, hidden=hidden)
    elif args.model == "EFIN":
        return EFIN(
            cate_dims, num_count,
            treatment_dim=1,   # binary scalar treatment
            hidden_dim=hidden,
            use_ziln=use_ziln,
        )
    else:
        raise ValueError(f"Unknown DNN model: {args.model}")


def run_dnn_experiment(seed, args, train_loader, val_loader, test_loader,
                       cate_dims, num_count, hparams: dict,
                       eval_loader=None):
    """
    eval_loader: loader used for metric computation after training.
      - Pass val_loader during Optuna inner trials (hyperparameter selection).
      - Pass test_loader (default) for final / untuned evaluation.
    """
    if eval_loader is None:
        eval_loader = test_loader

    set_seed(seed)
    print(f"\n>>> Training Model with Seed: {seed}")

    backbone = _build_dnn_backbone(args, cate_dims, num_count, hparams)

    # ── EFIN: self-contained model + dedicated trainer ─────────────────────
    if args.model == "EFIN":
        model = backbone
        trainer = EFINTrainer(
            model,
            lr=hparams["lr"],
            l2_reg=hparams["l2_reg"],
            epochs=hparams["epochs"],
            device=Config.DEVICE,
            lambda_c=hparams.get("lambda_c", 1.0),
            use_focal=args.use_focal,
            focal_gamma=hparams["focal_gamma"],
            focal_alpha=hparams["focal_alpha"],
            checkpoint_metric=getattr(args, "checkpoint_metric", "val_loss"),
        )
        print("Training...")
        history = trainer.train(train_loader, val_loader)
        best_val_loss = min(history["val_loss"]) if (val_loader is not None and "val_loss" in history) else 0.0
        print("Evaluating...")
        metrics = evaluate_dnn_model(model, eval_loader, device=Config.DEVICE)
        metrics["val_loss"] = best_val_loss
        return metrics

    if args.use_gating or args.use_ranking:
        model = VALOR(backbone, use_gating=args.use_gating)
        trainer = VALORTrainer(
            model,
            lr=hparams["lr"],
            l2_reg=hparams["l2_reg"],
            epochs=hparams["epochs"],
            device=Config.DEVICE,
            gamma=hparams["focal_gamma"],
            alpha=hparams["focal_alpha"],
            lambda_rank=hparams["lambda_rank"],
            use_ranking=args.use_ranking,
            checkpoint_metric=getattr(args, "checkpoint_metric", "val_loss"),
        )
    else:
        lam_lu   = hparams.get("lambda_lu", 0.0)   if getattr(args, "use_uplift_ranking", False) else 0.0
        lam_rank = hparams.get("lambda_rank", 0.0) if getattr(args, "use_response_ranking", False) else 0.0
        
        model = backbone
        trainer = BaselineTrainer(
            model,
            lr=hparams["lr"],
            l2_reg=hparams["l2_reg"],
            epochs=hparams["epochs"],
            device=Config.DEVICE,
            use_focal=args.use_focal,
            focal_gamma=hparams["focal_gamma"],
            focal_alpha=hparams["focal_alpha"],
            lambda_ipm=hparams["lambda_ipm"],
            lambda_prop=hparams["lambda_prop"],
            lambda_lu=lam_lu,
            lambda_resrank=lam_rank,
            checkpoint_metric=getattr(args, "checkpoint_metric", "val_loss"),
        )

    print("Training...")
    history = trainer.train(train_loader, val_loader)
    best_val_loss = min(history["val_loss"]) if (val_loader is not None and "val_loss" in history) else 0.0

    print("Evaluating...")
    metrics = evaluate_dnn_model(model, eval_loader, device=Config.DEVICE)
    metrics["val_loss"] = best_val_loss
    return metrics


def run_rerum_experiment(seed, args, train_loader, val_loader, test_loader,
                         cate_dims, num_count, hparams: dict,
                         eval_loader=None):
    """
    eval_loader: loader used for metric computation after training.
      - Pass val_loader during Optuna inner trials (hyperparameter selection).
      - Pass test_loader (default) for final / untuned evaluation.
    """
    if eval_loader is None:
        eval_loader = test_loader

    set_seed(seed)
    print(f"\n>>> Training RERUM Model with Seed: {seed}")

    # RERUM backbones do NOT need use_ziln — the wrapper adds its own ZILN heads
    backbone = _build_dnn_backbone(args, cate_dims, num_count, hparams, use_ziln=False)

    model = RERUMWrapper(backbone, outcome_hidden=hparams["hidden_dim"] // 2)
    lam_rank = hparams["lambda_rank"] if args.use_response_ranking else 0.0
    lam_lu   = hparams["lambda_lu"]   if args.use_uplift_ranking    else 0.0

    trainer = RERUMTrainer(
        model,
        lr=hparams["lr"],
        l2_reg=hparams["l2_reg"],
        epochs=hparams["epochs"],
        device=Config.DEVICE,
        lambda_rank=lam_rank,
        lambda_lu=lam_lu,
        lambda_ipm=hparams["lambda_ipm"],
        checkpoint_metric=getattr(args, "checkpoint_metric", "val_loss"),
    )

    print("Training...")
    history = trainer.train(train_loader, val_loader)
    best_val_loss = min(history["val_loss"]) if (val_loader is not None and "val_loss" in history) else 0.0

    print("Evaluating...")
    metrics = evaluate_dnn_model(model, eval_loader, device=Config.DEVICE)
    metrics["val_loss"] = best_val_loss
    return metrics


def run_tree_experiment(seed, args, X_train, t_train, y_train, X_test, t_test, y_test):
    set_seed(seed)
    print(f"\n>>> Training Tree Model with Seed: {seed}")

    if args.model == "CausalForest":
        model = CausalForestWrapper(random_state=seed)
    elif args.model == "ZILN-GBDT":
        model = ZILNGBDTForest(random_state=seed)
    else:
        raise ValueError(f"Unknown Tree model: {args.model}")

    print("Training...")
    model.fit(X_train, t_train, y_train)
    print("Evaluating...")
    metrics = evaluate_tree_model(model, X_test, t_test, y_test)
    # Trees don't output a standard valid loss; use negative AUUC as a proxy for minimization
    metrics["val_loss"] = -metrics["auuc"]
    return metrics


# =====================================================================
#  Core single-experiment runner (called by main() and tune.py)
# =====================================================================

def run_single_experiment(seed: int, args, hparams: dict,
                          train_loader=None, val_loader=None, test_loader=None,
                          cate_dims=None, num_count=None,
                          X_train=None, t_train=None, y_train=None,
                          X_test=None, t_test=None, y_test=None,
                          eval_loader=None):
    """
    Run one seed's worth of training + evaluation.

    eval_loader controls which split is used for metric computation:
      - None (default)  → uses test_loader  (CLI / final evaluation)
      - val_loader      → used by Optuna inner trials to avoid test-set leakage

    Routing logic:
      - Tree models              → run_tree_experiment
      - --rerum / any ranking    → run_rerum_experiment (ZILN via RERUMWrapper)
      - Everything else          → run_dnn_experiment

    Returns a metrics dict: {auuc, qini, lift_30, krcc, latency_ms}.
    """
    is_tree = args.model in ["CausalForest", "ZILN-GBDT"]
    if is_tree:
        return run_tree_experiment(seed, args, X_train, t_train, y_train,
                                   X_test, t_test, y_test)
    elif args.rerum:
        return run_rerum_experiment(seed, args, train_loader, val_loader,
                                    test_loader, cate_dims, num_count, hparams,
                                    eval_loader=eval_loader)
    else:
        return run_dnn_experiment(seed, args, train_loader, val_loader,
                                  test_loader, cate_dims, num_count, hparams,
                                  eval_loader=eval_loader)


# =====================================================================
#  Naming / saving helpers
# =====================================================================

def get_model_name(args):
    """Constructs a descriptive model name based on active flags."""
    name = args.model
    flags = []
    
    is_rerum = getattr(args, 'rerum', False)
    has_ziln = getattr(args, 'use_ziln', False)
    has_uplift = getattr(args, 'use_uplift_ranking', False)
    has_resp = getattr(args, 'use_response_ranking', False)
    has_focal = getattr(args, 'use_focal', False)
    has_gti = getattr(args, 'use_gating', False)
    has_wr = getattr(args, 'use_ranking', False)
    
    if is_rerum and has_ziln and has_uplift and has_resp:
        flags.append("RERUM")
        if has_focal: flags.append("Focal")
        if has_gti: flags.append("GTI")
        if has_wr: flags.append("WR")
    else:
        if has_ziln:
            flags.append("ZILN")
        if has_focal:
            flags.append("Focal")
        if has_gti:
            flags.append("GTI")
        if has_wr:
            flags.append("WR")
        if has_uplift:
            flags.append("UpliftRank")
        if has_resp:
            flags.append("RespRank")

    if flags:
        name += " + " + " + ".join(flags)
    return name


def get_project_root():
    """Helper to get the root directory path for saving results."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_to_csv(model_name, dataset_name, all_metrics, filename):
    """Appends aggregated results to a CSV file."""
    row = {"Model": model_name}
    for k, v_list in all_metrics.items():
        if k != "val_loss":
            row[f"{k}_mean"] = np.mean(v_list)
            row[f"{k}_std"] = np.std(v_list)

    df_new = pd.DataFrame([row])

    # Define column order
    cols = [
        "Model",
        "auuc_mean", "auuc_std",
        "qini_mean", "qini_std",
        "lift_30_mean", "lift_30_std",
        "krcc_mean", "krcc_std",
        "latency_ms_mean", "latency_ms_std"
    ]
    for c in cols:
        if c not in df_new.columns:
            df_new[c] = np.nan
    df_new = df_new[cols]

    root_dir = get_project_root()
    full_filename = os.path.join(root_dir, filename)
    os.makedirs(os.path.dirname(full_filename), exist_ok=True)

    if not os.path.exists(full_filename):
        df_new.to_csv(full_filename, index=False)
    else:
        df_new.to_csv(full_filename, mode='a', header=False, index=False)
    print(f"Results appended to {full_filename}")


def _get_output_path(args):
    """Return the results CSV path (relative to root) for untuned runs."""
    folder = "results/untuned_max_auqc" if getattr(args, "checkpoint_metric", "val_loss") == "val_qini" else "results/untuned"
    if args.dataset == "synthetic":
        return f"{folder}/synthesis.csv"
    elif args.dataset == "hillstrom-men":
        return f"{folder}/hillstrom_men.csv"
    elif args.dataset == "hillstrom-women":
        return f"{folder}/hillstrom_women.csv"
    else:
        return f"{folder}/{args.dataset}.csv"


# =====================================================================
#  Data loading helper
# =====================================================================

def load_data(args, hparams: dict):
    """
    Load and return all data structures needed for training.

    Returns a dict with keys:
      For DNN/RERUM:
        train_loader, val_loader, test_loader, cate_dims, num_count
        is_split_dataset
      For Tree models:
        X_train, t_train, y_train, X_test, t_test, y_test
      Always:
        df (full dataframe, for trees on synthetic)
        train_df, val_df, test_df (for split datasets, else None)
    """
    is_split_dataset = args.dataset in ["hillstrom-men", "hillstrom-women"]
    batch_size = hparams["batch_size"]

    train_df = val_df = test_df = df = None
    root_dir = get_project_root()

    if args.dataset == "synthetic":
        cache_path = os.path.join(root_dir, "dataset_cache.pkl")
        if os.path.exists(cache_path):
            print(f"📦 Loading cached dataset from {cache_path}...")
            df = pd.read_pickle(cache_path)
        else:
            df = create_synthetic_data(n_uid=Config.N_UID, n_pid=Config.N_PID, seed=42)
            print(f"💾 Saving dataset to cache {cache_path}...")
            df.to_pickle(cache_path)
    else:
        folder_name = "Men" if args.dataset == "hillstrom-men" else "Women"
        base_path = os.path.join(root_dir, f"dataset/Hillstrom/{folder_name}")
        prefix = "men" if args.dataset == "hillstrom-men" else "women"

        print(f"📦 Loading {args.dataset} from {base_path}...")
        train_df = pd.read_csv(f"{base_path}/train_{prefix}.csv")
        val_df   = pd.read_csv(f"{base_path}/val_{prefix}.csv")
        test_df  = pd.read_csv(f"{base_path}/test_{prefix}.csv")
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    result = {
        "df": df, "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "is_split_dataset": is_split_dataset,
    }

    is_tree = args.model in ["CausalForest", "ZILN-GBDT"]

    if is_tree:
        target_col = "label" if "label" in df.columns else "spend"
        if is_split_dataset:
            cat_cols, num_cols = _identify_columns(train_df)
            feature_cols = cat_cols + num_cols
            result["X_train"] = train_df[feature_cols].values
            result["t_train"] = train_df["treatment"].values
            result["y_train"] = train_df[target_col].values
            result["X_test"]  = test_df[feature_cols].values
            result["t_test"]  = test_df["treatment"].values
            result["y_test"]  = test_df[target_col].values
        else:
            cat_cols, num_cols = _identify_columns(df)
            feature_cols = cat_cols + num_cols
            X_all = df[feature_cols].values
            t_all = df["treatment"].values
            y_all = df[target_col].values

            N = len(df)
            rng = np.random.RandomState(42)
            perm = rng.permutation(N)
            n_test  = int(N * 0.1)
            n_val   = int(N * 0.2)
            n_train = N - n_test - n_val

            train_idx = perm[:n_train]
            test_idx  = perm[n_train + n_val:]

            result["X_train"] = X_all[train_idx]
            result["t_train"] = t_all[train_idx]
            result["y_train"] = y_all[train_idx]
            result["X_test"]  = X_all[test_idx]
            result["t_test"]  = t_all[test_idx]
            result["y_test"]  = y_all[test_idx]
    else:
        if is_split_dataset:
            loaders = get_dataloaders_from_splits(train_df, val_df, test_df,
                                                  batch_size=batch_size)
        else:
            loaders = get_dataloaders(df, batch_size=batch_size, seed=42)
        (result["train_loader"], result["val_loader"],
         result["test_loader"], result["cate_dims"], result["num_count"]) = loaders

    return result


# =====================================================================
#  CLI entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run VALOR experiments over fixed seeds with a fixed dataset."
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["TARNet", "DragonNet", "CFR-WASS", "CFR-MMD",
                                 "UniTE", "EUEN", "T-Learner", "S-Learner",
                                 "CausalForest", "ZILN-GBDT", "EFIN"],
                        help="Base model architecture to use.")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "hillstrom-men", "hillstrom-women"],
                        help="Dataset to run experiments on.")
    parser.add_argument("--rerum", action="store_true",
                        help="Wrap backbone with RERUM.")
    parser.add_argument("--use_ziln", action="store_true",
                        help="Enable ZILN head.")
    parser.add_argument("--use_focal", action="store_true",
                        help="Enable Focal-ZILN loss.")
    parser.add_argument("--use_gating", action="store_true",
                        help="Enable Treatment-Gated Interaction.")
    parser.add_argument("--use_ranking", action="store_true",
                        help="Enable Value-Weighted Ranking Loss (VALOR).")
    parser.add_argument("--use_uplift_ranking", action="store_true",
                        help="Enable Listwise Uplift Ranking Loss (RERUM).")
    parser.add_argument("--use_response_ranking", action="store_true",
                        help="Enable Pairwise Response Ranking Loss (RERUM).")
    parser.add_argument("--checkpoint_metric", "--checkpoint_restore", type=str, default="val_loss",
                        dest="checkpoint_metric",
                        choices=["val_loss", "val_qini"],
                        help="Metric to select best epoch checkpoint during training.")

    args = parser.parse_args()

    # Auto-enable ziln when required by other flags
    if (args.use_focal or args.use_gating or args.use_ranking) and not args.use_ziln:
        print("Warning: use_focal/use_gating/use_ranking require use_ziln. Enabling it.")
        args.use_ziln = True

    hparams = default_hparams()

    model_display_name = get_model_name(args)
    print(f"Experimental Setup: {model_display_name}")
    print(f"Dataset Seed: 42 (Fixed)")
    print(f"Model Seeds: {Config.SEEDS}")

    # 1. Load Data
    data = load_data(args, hparams)
    is_tree = args.model in ["CausalForest", "ZILN-GBDT"]

    # 2. Run Experiments
    all_metrics = defaultdict(list)

    for seed in Config.SEEDS:
        if is_tree:
            metrics = run_single_experiment(
                seed, args, hparams,
                X_train=data["X_train"], t_train=data["t_train"], y_train=data["y_train"],
                X_test=data["X_test"],   t_test=data["t_test"],   y_test=data["y_test"],
            )
        else:
            metrics = run_single_experiment(
                seed, args, hparams,
                train_loader=data["train_loader"],
                val_loader=data["val_loader"],
                test_loader=data["test_loader"],
                cate_dims=data["cate_dims"],
                num_count=data["num_count"],
            )
        for k, v in metrics.items():
            if v is not None:
                all_metrics[k].append(v)

    # 3. Aggregate Results
    print("\n" + "="*30)
    print(f"FINAL AGGREGATED RESULTS: {model_display_name}")
    print("="*30)
    for k, v_list in all_metrics.items():
        mean_val = np.mean(v_list)
        std_val  = np.std(v_list)
        if k == 'latency_ms':
            print(f"{k}: {mean_val:.4f} ± {std_val:.4f} ms")
        else:
            print(f"{k}: {mean_val:.4f} ± {std_val:.4f}")

    # 4. Save to results/untuned/
    out_filename = _get_output_path(args)
    save_to_csv(model_display_name, args.dataset, all_metrics, filename=out_filename)


if __name__ == "__main__":
    main()

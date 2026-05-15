import argparse
import sys
import os
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

from training.trainer import BaselineTrainer, VALORTrainer, RERUMTrainer
from training.evaluate import evaluate_dnn_model, evaluate_tree_model
from config import Config

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_dnn_experiment(seed, args, train_loader, val_loader, test_loader, cate_dims, num_count):
    set_seed(seed)
    print(f"\n>>> Training Model with Seed: {seed}")
    
    if args.model == "TARNet":
        backbone = TARNet(cate_dims, num_count, hidden=Config.HIDDEN_DIM, use_ziln=args.use_ziln)
    elif args.model == "DragonNet":
        backbone = DragonNet(cate_dims, num_count, hidden=Config.HIDDEN_DIM, use_ziln=args.use_ziln)
    elif args.model == "CFR-WASS":
        backbone = CFR(cate_dims, num_count, hidden=Config.HIDDEN_DIM, mode="wass", use_ziln=args.use_ziln)
    elif args.model == "CFR-MMD":
        backbone = CFR(cate_dims, num_count, hidden=Config.HIDDEN_DIM, mode="mmd", use_ziln=args.use_ziln)
    elif args.model == "UniTE":
        backbone = UniTE(cate_dims, num_count, hidden=Config.HIDDEN_DIM, use_ziln=args.use_ziln)
    elif args.model == "EUEN":
        backbone = EUEN(cate_dims, num_count, hidden=Config.HIDDEN_DIM, use_ziln=args.use_ziln)
    elif args.model == "T-Learner":
        backbone = TLearner(cate_dims, num_count, hidden=Config.HIDDEN_DIM, use_ziln=args.use_ziln)
    elif args.model == "S-Learner":
        backbone = SLearner(cate_dims, num_count, hidden=Config.HIDDEN_DIM)
    else:
        raise ValueError(f"Unknown DNN model: {args.model}")

    if args.use_gating or args.use_ranking:
        model = VALOR(backbone, use_gating=args.use_gating)
        trainer = VALORTrainer(
            model, lr=Config.LR, epochs=Config.EPOCHS, device=Config.DEVICE,
            gamma=Config.FOCAL_GAMMA, alpha=Config.FOCAL_ALPHA,
            lambda_rank=Config.LAMBDA_RANK, use_ranking=args.use_ranking
        )
    else:
        model = backbone
        trainer = BaselineTrainer(
            model, lr=Config.LR, epochs=Config.EPOCHS, device=Config.DEVICE,
            use_focal=args.use_focal, focal_gamma=Config.FOCAL_GAMMA,
            focal_alpha=Config.FOCAL_ALPHA, lambda_ipm=Config.LAMBDA_IPM,
            lambda_prop=Config.LAMBDA_PROP
        )

    print("Training...")
    trainer.train(train_loader, val_loader)
    
    print("Evaluating...")
    metrics = evaluate_dnn_model(model, test_loader, device=Config.DEVICE)
    return metrics


def run_rerum_experiment(seed, args, train_loader, val_loader, test_loader, cate_dims, num_count):
    set_seed(seed)
    print(f"\n>>> Training RERUM Model with Seed: {seed}")

    # Backbones for RERUM do NOT need use_ziln — the wrapper adds its own ZILN heads
    if args.model == "TARNet":
        backbone = TARNet(cate_dims, num_count, hidden=Config.HIDDEN_DIM)
    elif args.model == "DragonNet":
        backbone = DragonNet(cate_dims, num_count, hidden=Config.HIDDEN_DIM)
    elif args.model == "CFR-WASS":
        backbone = CFR(cate_dims, num_count, hidden=Config.HIDDEN_DIM, mode="wass")
    elif args.model == "CFR-MMD":
        backbone = CFR(cate_dims, num_count, hidden=Config.HIDDEN_DIM, mode="mmd")
    elif args.model == "UniTE":
        backbone = UniTE(cate_dims, num_count, hidden=Config.HIDDEN_DIM)
    elif args.model == "EUEN":
        backbone = EUEN(cate_dims, num_count, hidden=Config.HIDDEN_DIM)
    elif args.model == "T-Learner":
        backbone = TLearner(cate_dims, num_count, hidden=Config.HIDDEN_DIM)
    else:
        raise ValueError(f"RERUM does not support model: {args.model}")

    model = RERUMWrapper(backbone, outcome_hidden=Config.HIDDEN_DIM // 2)
    lam_rank = Config.LAMBDA_RANK if args.use_response_ranking else 0.0
    lam_lu = Config.LAMBDA_LU if args.use_uplift_ranking else 0.0
    
    # If neither are set but rerun is True, RERUM implies both by default, 
    # but since the user wants explicit flags, we use the flags directly.
    # We default them to True if not provided to keep backward compatibility 
    # if someone runs --rerum without flags.
    # Wait, argparse sets them to False if not passed, because of action="store_true".
    # So if the user explicitly passes --rerum, they MUST pass the flags, or they get ZILN only.
    # To keep default full RERUM: if both are false, but rerum is true, maybe they meant full RERUM?
    # Let's just strictly follow the flags. If they want full RERUM, they use both flags.
    
    trainer = RERUMTrainer(
        model, lr=Config.LR, epochs=Config.EPOCHS, device=Config.DEVICE,
        lambda_rank=lam_rank, lambda_lu=lam_lu,
        lambda_ipm=Config.LAMBDA_IPM,
    )

    print("Training...")
    trainer.train(train_loader, val_loader)

    print("Evaluating...")
    metrics = evaluate_dnn_model(model, test_loader, device=Config.DEVICE)
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
    return metrics

def get_model_name(args):
    """Constructs a descriptive model name based on active flags."""
    name = args.model
    flags = []
    if args.rerum:
        flags.append("RERUM")
    if args.use_ziln:
        flags.append("ZILN")
    if args.use_focal:
        flags.append("Focal")
    if args.use_gating:
        flags.append("GTI")
    if getattr(args, 'use_ranking', False):
        flags.append("WR")
    if getattr(args, 'use_uplift_ranking', False):
        flags.append("UpliftRank")
    if getattr(args, 'use_response_ranking', False):
        flags.append("RespRank")
    
    if flags:
        name += " + " + " + ".join(flags)
    return name

def save_to_csv(model_name, dataset_name, all_metrics, filename="result2.csv"):
    """Appends aggregated results to a CSV file."""
    row = {"Model": model_name}
    for k, v_list in all_metrics.items():
        row[k] = np.mean(v_list)
    
    df_new = pd.DataFrame([row])
    
    # Define column order
    cols = ["Model", "auuc", "qini", "lift_30", "krcc", "latency_ms"]
    # Ensure all columns exist in df_new
    for c in cols:
        if c not in df_new.columns:
            df_new[c] = np.nan
    df_new = df_new[cols]
    
    if not os.path.exists(filename):
        df_new.to_csv(filename, index=False)
    else:
        # Append without header
        df_new.to_csv(filename, mode='a', header=False, index=False)
    print(f"Results appended to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run VALOR experiments over 5 fixed seeds with a fixed dataset.")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["TARNet", "DragonNet", "CFR-WASS", "CFR-MMD", "UniTE", "EUEN", 
                                 "T-Learner", "S-Learner", "CausalForest", "ZILN-GBDT"],
                        help="Base model architecture to use.")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "hillstrom-men", "hillstrom-women"],
                        help="Dataset to run experiments on.")
    parser.add_argument("--rerum", action="store_true",
                        help="Wrap backbone with RERUM (ZILN + pairwise + listwise losses).")
    parser.add_argument("--use_ziln", action="store_true", help="Enable ZILN head (RERUM mode).")
    parser.add_argument("--use_focal", action="store_true", help="Enable Focal-ZILN loss (requires --use_ziln).")
    parser.add_argument("--use_gating", action="store_true", help="Enable Treatment-Gated Interaction (requires --use_ziln).")
    parser.add_argument("--use_ranking", action="store_true", help="Enable Value-Weighted Ranking Loss (VALOR mode).")
    parser.add_argument("--use_uplift_ranking", action="store_true", help="Enable Listwise Uplift Ranking Loss (RERUM mode).")
    parser.add_argument("--use_response_ranking", action="store_true", help="Enable Pairwise Response Ranking Loss (RERUM mode).")
    
    args = parser.parse_args()

    # validations
    if (args.use_focal or args.use_gating or args.use_ranking) and not args.use_ziln:
        print("Warning: use_focal, use_gating, and use_ranking require use_ziln=True. Enabling use_ziln.")
        args.use_ziln = True

    model_display_name = get_model_name(args)
    print(f"Experimental Setup: {model_display_name}")
    print(f"Dataset Seed: 42 (Fixed)")
    print(f"Model Seeds: {Config.SEEDS}")

    # 1. Load Data
    is_split_dataset = args.dataset in ["hillstrom-men", "hillstrom-women"]
    
    if args.dataset == "synthetic":
        cache_path = "dataset_cache.pkl"
        if os.path.exists(cache_path):
            print(f"📦 Loading cached dataset from {cache_path}...")
            df = pd.read_pickle(cache_path)
        else:
            df = create_synthetic_data(n_uid=Config.N_UID, n_pid=Config.N_PID, seed=42)
            print(f"💾 Saving dataset to cache {cache_path}...")
            df.to_pickle(cache_path)
    else:
        # Load Hillstrom splits
        folder_name = "Men" if args.dataset == "hillstrom-men" else "Women"
        base_path = f"dataset/Hillstrom/{folder_name}"
        prefix = "men" if args.dataset == "hillstrom-men" else "women"
        
        print(f"📦 Loading {args.dataset} from {base_path}...")
        train_df = pd.read_csv(f"{base_path}/train_{prefix}.csv")
        val_df = pd.read_csv(f"{base_path}/val_{prefix}.csv")
        test_df = pd.read_csv(f"{base_path}/test_{prefix}.csv")
        df = pd.concat([train_df, val_df, test_df], ignore_index=True) # Full dataframe for tree models if needed
    all_metrics = defaultdict(list)

    # 2. Run Experiments
    if args.model in ["CausalForest", "ZILN-GBDT"]:
        # Prepare Tree Data Once (Fixed Split Seed 42)
        target_col = "label" if "label" in df.columns else "spend"
        
        if is_split_dataset:
            cat_cols, num_cols = _identify_columns(train_df)
            feature_cols = cat_cols + num_cols
            X_train, t_train, y_train = train_df[feature_cols].values, train_df["treatment"].values, train_df[target_col].values
            X_test, t_test, y_test = test_df[feature_cols].values, test_df["treatment"].values, test_df[target_col].values
        else:
            cat_cols, num_cols = _identify_columns(df)
            feature_cols = cat_cols + num_cols
            X_all = df[feature_cols].values
            t_all = df["treatment"].values
            y_all = df[target_col].values

            N = len(df)
            rng = np.random.RandomState(42)
            perm = rng.permutation(N)
            n_test = int(N * 0.1)
            n_val = int(N * 0.2)
            n_train = N - n_test - n_val

            train_idx = perm[:n_train]
            test_idx = perm[n_train + n_val:]

            X_train, t_train, y_train = X_all[train_idx], t_all[train_idx], y_all[train_idx]
            X_test, t_test, y_test = X_all[test_idx], t_all[test_idx], y_all[test_idx]

        for seed in Config.SEEDS:
            metrics = run_tree_experiment(seed, args, X_train, t_train, y_train, X_test, t_test, y_test)
            for k, v in metrics.items():
                if v is not None:
                    all_metrics[k].append(v)
    else:
        # Prepare DNN Data Once (Fixed Split Seed 42)
        if is_split_dataset:
            train_loader, val_loader, test_loader, cate_dims, num_count = get_dataloaders_from_splits(
                train_df, val_df, test_df, batch_size=Config.BATCH_SIZE
            )
        else:
            train_loader, val_loader, test_loader, cate_dims, num_count = get_dataloaders(
                df, batch_size=Config.BATCH_SIZE, seed=42
            )

        for seed in Config.SEEDS:
            if args.rerum:
                metrics = run_rerum_experiment(seed, args, train_loader, val_loader, test_loader, cate_dims, num_count)
            else:
                metrics = run_dnn_experiment(seed, args, train_loader, val_loader, test_loader, cate_dims, num_count)
            for k, v in metrics.items():
                if v is not None:
                    all_metrics[k].append(v)

    # 3. Aggregate Results
    print("\n" + "="*30)
    print(f"FINAL AGGREGATED RESULTS: {model_display_name}")
    print("="*30)
    for k, v_list in all_metrics.items():
        mean_val = np.mean(v_list)
        std_val = np.std(v_list)
        if k == 'latency_ms':
            print(f"{k}: {mean_val:.4f} ± {std_val:.4f} ms")
        else:
            print(f"{k}: {mean_val:.4f} ± {std_val:.4f}")

    # 4. Save
    if args.dataset == "synthetic":
        out_filename = "result/synthesis.csv"
    elif args.dataset == "hillstrom-men":
        out_filename = "result/hillstrom_men.csv"
    elif args.dataset == "hillstrom-women":
        out_filename = "result/hillstrom_women.csv"
    else:
        out_filename = f"result/{args.dataset}.csv"
        
    save_to_csv(model_display_name, args.dataset, all_metrics, filename=out_filename)

if __name__ == "__main__":
    main()

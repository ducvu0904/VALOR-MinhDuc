import argparse
import sys
import os
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

from dataset.generate import create_synthetic_data
from dataset.dataloader import get_dataloaders, _identify_columns

from models.baselines import TARNet, DragonNet, CFR, UniTE, EUEN, TLearner, SLearner, CausalForestWrapper
from models.valor_net import VALOR
from models.ziln_gbdt import ZILNGBDTForest

from training.trainer import BaselineTrainer, VALORTrainer
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
    if args.use_ziln:
        flags.append("ZILN")
    if args.use_focal:
        flags.append("Focal")
    if args.use_gating:
        flags.append("GTI")
    if args.use_ranking:
        flags.append("WR")
    
    if flags:
        name += " + " + " + ".join(flags)
    return name

def save_to_csv(model_name, all_metrics, filename="results.csv"):
    """Appends aggregated results to a CSV file."""
    row = {"Model": model_name}
    for k, v_list in all_metrics.items():
        row[k] = np.mean(v_list)
    
    df_new = pd.DataFrame([row])
    
    # Define column order (Model, AUUC, Qini/AUQC, Lift@30, KRCC, Latency)
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
    parser.add_argument("--use_ziln", action="store_true", help="Enable ZILN head (RERUM mode).")
    parser.add_argument("--use_focal", action="store_true", help="Enable Focal-ZILN loss (requires --use_ziln).")
    parser.add_argument("--use_gating", action="store_true", help="Enable Treatment-Gated Interaction (requires --use_ziln).")
    parser.add_argument("--use_ranking", action="store_true", help="Enable Value-Weighted Ranking Loss (requires --use_ziln).")
    
    args = parser.parse_args()

    # validations
    if (args.use_focal or args.use_gating or args.use_ranking) and not args.use_ziln:
        print("Warning: use_focal, use_gating, and use_ranking require use_ziln=True. Enabling use_ziln.")
        args.use_ziln = True

    model_display_name = get_model_name(args)
    print(f"Experimental Setup: {model_display_name}")
    print(f"Dataset Seed: 42 (Fixed)")
    print(f"Model Seeds: {Config.SEEDS}")

    # 1. Generate Data Once
    df = create_synthetic_data(n_uid=Config.N_UID, n_pid=Config.N_PID, seed=42)
    all_metrics = defaultdict(list)

    # 2. Run Experiments
    if args.model in ["CausalForest", "ZILN-GBDT"]:
        # Prepare Tree Data Once (Fixed Split Seed 42)
        cat_cols, num_cols = _identify_columns(df)
        feature_cols = cat_cols + num_cols
        X_all = df[feature_cols].values
        t_all = df["treatment"].values
        y_all = df["label"].values

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
        train_loader, val_loader, test_loader, cate_dims, num_count = get_dataloaders(
            df, batch_size=Config.BATCH_SIZE, seed=42
        )

        for seed in Config.SEEDS:
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

    # 4. Save to CSV
    save_to_csv(model_display_name, all_metrics)

if __name__ == "__main__":
    main()

import argparse
import sys
import os
import numpy as np
import torch

from dataset.generate import create_synthetic_data
from dataset.dataloader import get_dataloaders, _identify_columns

from models.baselines import TARNet, DragonNet, CFR, UniTE, EUEN, TLearner, SLearner, CausalForestWrapper
from models.valor_net import VALOR
from models.ziln_gbdt import ZILNGBDTForest

from training.trainer import BaselineTrainer, VALORTrainer
from training.evaluate import evaluate_dnn_model, evaluate_tree_model
from config import Config

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Run VALOR experiments for a specific model.")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["TARNet", "DragonNet", "CFR-WASS", "CFR-MMD", "UniTE", "EUEN", 
                                 "T-Learner", "S-Learner", "CausalForest", "ZILN-GBDT"],
                        help="Base model architecture to use.")
    parser.add_argument("--use_ziln", action="store_true", help="Enable ZILN head (RERUM mode).")
    parser.add_argument("--use_focal", action="store_true", help="Enable Focal-ZILN loss (requires --use_ziln).")
    parser.add_argument("--use_gating", action="store_true", help="Enable Treatment-Gated Interaction (requires --use_ziln).")
    parser.add_argument("--use_ranking", action="store_true", help="Enable Value-Weighted Ranking Loss (requires --use_ziln).")
    parser.add_argument("--seed", type=int, default=Config.SEEDS[0], help="Random seed.")
    
    args = parser.parse_args()

    # validations
    if (args.use_focal or args.use_gating or args.use_ranking) and not args.use_ziln:
        print("Warning: use_focal, use_gating, and use_ranking require use_ziln=True. Enabling use_ziln.")
        args.use_ziln = True

    print(f"Running {args.model} with seed {args.seed}")
    print(f"Config: ZILN={args.use_ziln}, Focal={args.use_focal}, Gating={args.use_gating}, Ranking={args.use_ranking}")
    
    set_seed(args.seed)

    df = create_synthetic_data(n_uid=Config.N_UID, n_pid=Config.N_PID, seed=args.seed)
    
    # Trees
    if args.model in ["CausalForest", "ZILN-GBDT"]:
        cat_cols, num_cols = _identify_columns(df)
        feature_cols = cat_cols + num_cols
        X_all = df[feature_cols].values
        t_all = df["treatment"].values
        y_all = df["label"].values

        N = len(df)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(N)
        n_test = int(N * 0.1)
        n_val = int(N * 0.2)
        n_train = N - n_test - n_val

        train_idx = perm[:n_train]
        test_idx = perm[n_train + n_val:]

        X_train, t_train, y_train = X_all[train_idx], t_all[train_idx], y_all[train_idx]
        X_test, t_test, y_test = X_all[test_idx], t_all[test_idx], y_all[test_idx]

        if args.model == "CausalForest":
            model = CausalForestWrapper(random_state=args.seed)
        else:
            model = ZILNGBDTForest(random_state=args.seed)
            
        print("Training...")
        model.fit(X_train, t_train, y_train)
        print("Evaluating...")
        metrics = evaluate_tree_model(model, X_test, t_test, y_test)
        print(f"\nResults:\nAUUC: {metrics['auuc']:.4f}\nQini: {metrics['qini']:.4f}\nLift@30: {metrics['lift_30']:.4f}\nKRCC: {metrics['krcc']:.4f}\nLatency: {metrics['latency_ms']:.4f} ms")
        return

    # DNN
    train_loader, val_loader, test_loader, cate_dims, num_count = get_dataloaders(
        df, batch_size=Config.BATCH_SIZE, seed=args.seed
    )

    if args.model == "TARNet":
        backbone = TARNet(cate_dims, num_count, use_ziln=args.use_ziln)
    elif args.model == "DragonNet":
        backbone = DragonNet(cate_dims, num_count, use_ziln=args.use_ziln)
    elif args.model == "CFR-WASS":
        backbone = CFR(cate_dims, num_count, mode="wass", use_ziln=args.use_ziln)
    elif args.model == "CFR-MMD":
        backbone = CFR(cate_dims, num_count, mode="mmd", use_ziln=args.use_ziln)
    elif args.model == "UniTE":
        backbone = UniTE(cate_dims, num_count, use_ziln=args.use_ziln)
    elif args.model == "EUEN":
        backbone = EUEN(cate_dims, num_count, use_ziln=args.use_ziln)
    elif args.model == "T-Learner":
        backbone = TLearner(cate_dims, num_count, use_ziln=args.use_ziln)
    elif args.model == "S-Learner":
        backbone = SLearner(cate_dims, num_count)

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
    print(f"\nResults:\nAUUC: {metrics['auuc']:.4f}\nQini: {metrics['qini']:.4f}\nLift@30: {metrics['lift_30']:.4f}\nKRCC: {metrics['krcc']:.4f}\nLatency: {metrics['latency_ms']:.4f} ms")

if __name__ == "__main__":
    main()

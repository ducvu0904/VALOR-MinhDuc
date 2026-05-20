import torch


class Config:
    # Data generation
    N_UID = 5000
    N_PID = 500000

    # Training Hyperparameters
    SEEDS = [42, 123, 456, 789, 1024]
    EPOCHS = 30
    LR = 5e-4
    BATCH_SIZE = 512
    HIDDEN_DIM = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss Hyperparameters
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.9
    LAMBDA_RANK = 1e-4
    LAMBDA_LU   = 0.01
    LAMBDA_IPM  = 1.0
    LAMBDA_PROP = 1.0


def default_hparams() -> dict:
    """
    Return the canonical set of tunable hyperparameters as a dictionary.

    This dict is the standard interface used by:
      - main.py  (uses defaults)
      - tune.py  (Optuna overrides individual keys)
      - run_all.py (calls main.py which uses defaults)

    All trainer / model constructors must accept these keys so that
    tune.py can override any subset without touching other code.
    """
    return {
        # Optimisation
        "lr":          Config.LR,
        "l2_reg":      1e-5,
        "batch_size":  Config.BATCH_SIZE,

        # Architecture
        "hidden_dim":  Config.HIDDEN_DIM,

        # Focal loss
        "focal_gamma": Config.FOCAL_GAMMA,
        "focal_alpha": Config.FOCAL_ALPHA,

        # Ranking / auxiliary loss weights
        "lambda_rank": Config.LAMBDA_RANK,
        "lambda_lu":   Config.LAMBDA_LU,
        "lambda_ipm":  Config.LAMBDA_IPM,
        "lambda_prop": Config.LAMBDA_PROP,

        # Training schedule
        "epochs":      Config.EPOCHS,
    }

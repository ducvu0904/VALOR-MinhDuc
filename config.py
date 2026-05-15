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

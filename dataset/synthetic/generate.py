"""Synthetic data generation logic for Hybrid UMLC-VALOR experiments.

This module provides functions to generate synthetic datasets using the exact 
feature space from UMLC and the ZILN (Zero-Inflated Log-Normal) response 
distribution from VALOR.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def set_random_seed(seed: int):
    """Sets random seeds for reproducibility."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_dataframe(
    n_samples: int,
    n_binary: int,
    n_continuous: int,
    n_categorical: int,
    n_classes_categorical: int,
    str_name: str,
    cluster: bool = False
) -> pd.DataFrame:
    """Generates a base dataframe matching UMLC feature specifications."""
    columns = []
    data = []

    if n_binary > 0:
        binary_data = np.random.binomial(1, 0.5, (n_samples, n_binary))
        binary_columns = [f'{str_name}_bin_{i+1}' for i in range(n_binary)]
        columns.extend(binary_columns)
        data.append(binary_data)

    if n_continuous > 0:
        continuous_data = np.random.normal(0, 1, (n_samples, n_continuous))
        continuous_columns = [f'{str_name}_cont_{i+1}' for i in range(n_continuous)]
        columns.extend(continuous_columns)
        data.append(continuous_data)

    if n_categorical > 0:
        cat_data = np.random.randint(0, n_classes_categorical, (n_samples, n_categorical))
        cat_columns = [f'{str_name}_cat_{i+1}' for i in range(n_categorical)]
        columns.extend(cat_columns)
        data.append(cat_data)

    if cluster:
        # Corrected: Applied the exact multinomial probabilities from the UMLC paper
        cluster_probs = [0.20, 0.16, 0.16, 0.16, 0.16, 0.16]
        cluster_labels = np.random.choice(6, size=n_samples, p=cluster_probs)
        columns.append('data_cluster')
        data.append(cluster_labels.reshape(-1, 1))

    df = pd.DataFrame(np.column_stack(data), columns=columns)
    return df


import numpy as np
import pandas as pd

def create_synthetic_data(n_uid: int = 20000, n_pid: int = 2000000, seed: int = 42) -> pd.DataFrame:
    print("🚀 Generating Optimized Hybrid Data (UMLC Features + VALOR ZILN)...")
    set_random_seed(seed)

    # 1. Create uid and pid
    df_uid = generate_dataframe(n_samples=n_uid, n_binary=34, n_continuous=66, n_categorical=0, n_classes_categorical=0, str_name='uid', cluster=False)
    df_pid = generate_dataframe(n_samples=n_pid, n_binary=34, n_continuous=66, 
                                n_categorical=3, n_classes_categorical=4, str_name='pid', cluster=True)

    # Assign treatment at user level
    df_uid['treatment'] = np.random.binomial(1, 0.5, n_uid)

    # Convert to NumPy for high-speed processing
    uid_values = df_uid.values
    pid_values = df_pid.values
    uid_cols = df_uid.columns.tolist()
    pid_cols = df_pid.columns.tolist()
    
    # 2. Optimized Memory Join Logic
    all_rows = []
    for i in range(n_uid):
        n_context = np.random.randint(60, 131) # UMLC Spec: 60-130 
        pid_idx = np.random.choice(n_pid, size=n_context, replace=False)
        
        # Tile user row và ghép với selected pids
        u_row = uid_values[i:i+1]
        u_repeated = np.repeat(u_row, n_context, axis=0)
        joined_block = np.concatenate([u_repeated, pid_values[pid_idx]], axis=1)
        all_rows.append(joined_block)

    # Concatenate once
    result_df = pd.DataFrame(np.vstack(all_rows), columns=uid_cols + pid_cols)
    
    # Convert back to int for cluster/treatment columns (vstack converts everything to float)
    result_df['data_cluster'] = result_df['data_cluster'].astype(int)
    result_df['treatment'] = result_df['treatment'].astype(int)

    # 3. Calculate Scalars & Heterogeneity
    u_cols = [c for c in uid_cols if 'bin' in c or 'cont' in c]
    p_cols = [c for c in pid_cols if 'bin' in c or 'cont' in c]
    
    u_sum = result_df[u_cols].sum(axis=1)
    p_sum = result_df[p_cols].sum(axis=1)
    interaction = u_sum * p_sum
    p_cat_sum = result_df[[c for c in pid_cols if 'cat' in c]].sum(axis=1)

    # Create z0 and z1 independently to avoid collinearity (Create true HTE)
    cluster_bias_map = {
        0: (0, 1), 1: (2, 0.5), 2: (-1, 2), 3: (3, 1.5), 4: (-2, 0.8), 5: (1, 2)
    }
    z0 = np.zeros(len(result_df))
    z1 = np.zeros(len(result_df))
    
    for cid, (mu, std) in cluster_bias_map.items():
        mask = result_df['data_cluster'] == cid
        count = mask.sum()
        if count > 0:
            z0[mask] = np.random.normal(mu, std, count)
            z1[mask] = np.random.normal(mu, std, count) # Noise mới cho z1

    def normalize(x): return (x - x.mean()) / (x.std() + 1e-5)

    # Base and Treatment Effect independently
    f_base = normalize(0.5 * (u_sum + p_sum + interaction + p_cat_sum) + z0)
    f_te   = normalize(0.2 * (u_sum + p_sum + interaction + p_cat_sum) + z1)

    # 4. VALOR ZILN Distribution
    # Sparsity > 80% for B2B
    prob_c = 1 / (1 + np.exp(-(-2 + 0.5 * f_base)))
    prob_t = 1 / (1 + np.exp(-(-2 + 0.5 * f_base + 0.4 + 0.2 * f_te)))

    # Revenue Magnitude (Log-Normal)
    mu_c = 3.0 + 0.3 * f_base
    mu_t = mu_c + 0.2 + 0.2 * f_te
    sigma = 0.6 # Whales impact

    # Outcome Sampling
    conv_c = np.random.binomial(1, prob_c)
    conv_t = np.random.binomial(1, prob_t)
    
    result_df['y0'] = conv_c * np.exp(np.random.normal(mu_c, sigma))
    result_df['y1'] = conv_t * np.exp(np.random.normal(mu_t, sigma))
    
    # True Tau
    ev_c = prob_c * np.exp(mu_c + 0.5 * sigma**2)
    ev_t = prob_t * np.exp(mu_t + 0.5 * sigma**2)
    result_df['true_tau'] = ev_t - ev_c
    
    result_df['label'] = np.where(result_df['treatment'] == 1, result_df['y1'], result_df['y0'])

    print(f"✅ Data Generated. Shape: {result_df.shape}")
    print(f"📊 Sparsity: {(result_df['label'] == 0).mean():.2%}")
    return result_df
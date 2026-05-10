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


def create_synthetic_data(
    n_uid: int = 5000,
    n_pid: int = 50000,
    seed: int = 123
) -> pd.DataFrame:
    """Generates Hybrid Synthetic Data (UMLC Features + VALOR ZILN).

    Args:
        n_uid: Number of users.
        n_pid: Number of products.
        seed: Random seed.

    Returns:
        A DataFrame containing features, treatment, labels, and true tau.
    """
    print("Generating Synthetic Data (Hybrid UMLC Features + VALOR ZILN Response)...")
    set_random_seed(seed)

    # 1. Create User (UID) Data (UMLC Spec: 34 binary, 66 continuous)
    df_uid = generate_dataframe(
        n_samples=n_uid, n_binary=34, n_continuous=66,
        n_categorical=0, n_classes_categorical=0, str_name='uid'
    )

    # 2. Create Product (PID) Data (UMLC Spec: 34 bin, 66 cont, 3 cat (4 classes), 1 cluster)
    df_pid = generate_dataframe(
        n_samples=n_pid, n_binary=34, n_continuous=66,
        n_categorical=3, n_classes_categorical=4, str_name='pid', cluster=True
    )

    # Separate column lists for aggregation
    uid_cols = df_uid.columns.tolist()
    pid_bin_cont_cols = [c for c in df_pid.columns if 'bin' in c or 'cont' in c]
    pid_cat_cols = [c for c in df_pid.columns if 'cat' in c]

    # 3. Join logic (UMLC Spec: 60 to 130 context interactions per user)
    joined_data = []
    for _, row in df_uid.iterrows():
        n = np.random.randint(30,61)
        selected_rows = df_pid.sample(n=n, replace=True)
        repeated_row = pd.concat([pd.DataFrame([row])] * n, ignore_index=True)
        joined_row = pd.concat([repeated_row, selected_rows.reset_index(drop=True)], axis=1)
        joined_data.append(joined_row)

    result_df = pd.concat(joined_data, ignore_index=True)

    # 4. Feature Aggregation (Mapping UMLC to VALOR Scalars)
    u_id_sum = result_df[uid_cols].sum(axis=1)
    p_id_sum = result_df[pid_bin_cont_cols].sum(axis=1)
    p_cat_sum = result_df[pid_cat_cols].sum(axis=1)

    # Map cluster IDs to cluster bias distributions from UMLC paper
    cluster_bias_map = {
        0: lambda n: np.random.normal(0, 1, n),
        1: lambda n: np.random.normal(2, 0.5, n),
        2: lambda n: np.random.normal(-1, 2, n),
        3: lambda n: np.random.normal(3, 1.5, n),
        4: lambda n: np.random.normal(-2, 0.8, n),
        5: lambda n: np.random.normal(1, 2, n)
    }

    z_cluster = np.zeros(len(result_df))
    for cluster_id in range(6):
        mask = result_df['data_cluster'] == cluster_id
        count = mask.sum()
        if count > 0:
            z_cluster[mask] = cluster_bias_map[cluster_id](count)

    # Calculate Raw Scalars based on UMLC weightings
    raw_user = 0.5 * u_id_sum
    raw_item = 0.5 * p_id_sum + 0.5 * p_cat_sum + z_cluster
    raw_interaction = 0.5 * (u_id_sum * p_id_sum)

    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-5)

    feat_user = normalize(raw_user)
    feat_item = normalize(raw_item)
    feat_interaction = normalize(raw_interaction)

    # 5. Generate Outcome (ZILN Distribution)
    # --- A. Propensity (Gate) Generation ---
    # Baseline bias shifted to -2 to enforce >80% non-conversion sparsity
    base_logits = -2 + 0.5 * feat_user + 0.2 * feat_item
    prob_c = 1 / (1 + np.exp(-base_logits))
    prob_t = 1 / (1 + np.exp(-(base_logits + 0.3 + 0.1 * feat_interaction)))

    # --- B. Revenue (Mu) Generation ---
    base_mu = 3.0 + 0.3 * feat_interaction + 0.2 * feat_user
    mu_c = base_mu
    mu_t = base_mu + 0.2 + 0.2 * feat_interaction
    sigma = 0.5

    # --- C. Sample Outcomes ---
    conv_c = np.random.binomial(1, prob_c)
    conv_t = np.random.binomial(1, prob_t)
    rev_c = np.exp(np.random.normal(mu_c, sigma))
    rev_t = np.exp(np.random.normal(mu_t, sigma))

    y0 = conv_c * rev_c
    y1 = conv_t * rev_t

    # --- D. Calculate True Tau (Expected Value Difference) ---
    ev_c = prob_c * np.exp(mu_c + 0.5 * sigma**2)
    ev_t = prob_t * np.exp(mu_t + 0.5 * sigma**2)
    true_tau = ev_t - ev_c

    # Assign to DataFrame
    result_df['y0'] = y0
    result_df['y1'] = y1
    result_df['true_tau'] = true_tau
    result_df['treatment'] = np.random.choice([0, 1], size=len(result_df))
    result_df['label'] = np.where(result_df['treatment'] == 0, result_df['y0'], result_df['y1'])

    print(f"Data Generated. Shape: {result_df.shape}")
    print(f"Sparsity: {(result_df['label'] == 0).mean():.2%}")
    return result_df
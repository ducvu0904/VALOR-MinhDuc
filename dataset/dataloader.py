"""
DataLoader for the VALOR B2B-Mimic synthetic dataset.

Expected DataFrame columns (from ``create_synthetic_data()``):
 - ``uid_bin_*``    — binary UID features
 - ``uid_cont_*``   — continuous UID features
 - ``pid_bin_*``    — binary PID features
 - ``pid_cont_*``   — continuous PID features
 - ``pid_cat_*``    — categorical PID features  (embedded)
 - ``data_cluster`` — categorical cluster label  (embedded)
 - ``treatment``    — 0 / 1
 - ``label``        — observed outcome Y ≥ 0
 - ``y0``, ``y1``   — potential outcomes
 - ``true_tau``     — ground-truth CATE
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Columns that are treated as **categorical** (need Embedding layers)
CAT_COLUMNS = ["pid_cat_1", "pid_cat_2", "pid_cat_3", "data_cluster"]

# Columns that are NOT features
NON_FEATURE_COLS = {"y0", "y1", "true_tau", "treatment", "label", "spend", "conversion", "visit"}


def _identify_columns(df: pd.DataFrame):
    """Return (cat_cols, num_cols) lists for the given DataFrame."""
    cat_cols = [c for c in CAT_COLUMNS if c in df.columns]
    num_cols = [
        c for c in df.columns
        if c not in set(cat_cols) | NON_FEATURE_COLS
    ]
    return cat_cols, num_cols


def get_cate_dims(df: pd.DataFrame):
    """
    Return list of vocabulary sizes for each categorical column.
    Add 1 to max value to handle 0-indexed embeddings.
    """
    cat_cols, _ = _identify_columns(df)
    return [int(df[c].max()) + 1 for c in cat_cols]


class UpliftDataset(Dataset):
    """
    PyTorch Dataset for uplift modeling.

    Each item returns:
        x_cat     : LongTensor  (n_cat,)
        x_num     : FloatTensor (n_num,)
        treatment : FloatTensor scalar
        label     : FloatTensor scalar (observed outcome)
        true_tau  : FloatTensor scalar (ground-truth CATE)
    """

    def __init__(self, df: pd.DataFrame):
        cat_cols, num_cols = _identify_columns(df)

        self.x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.x_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.treatment = torch.tensor(df["treatment"].values, dtype=torch.float32)
        
        # Handle label mapping (synthetic vs Hillstrom)
        if "label" in df.columns:
            self.label = torch.tensor(df["label"].values, dtype=torch.float32)
        elif "spend" in df.columns:
            self.label = torch.tensor(df["spend"].values, dtype=torch.float32)
        else:
            raise KeyError("Dataset must contain either 'label' or 'spend' as the target variable.")

        # Handle missing true_tau (real-world datasets don't have this)
        if "true_tau" in df.columns:
            self.true_tau = torch.tensor(df["true_tau"].values, dtype=torch.float32)
        else:
            self.true_tau = torch.zeros_like(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.x_cat[idx],
            self.x_num[idx],
            self.treatment[idx],
            self.label[idx],
            self.true_tau[idx],
        )


def get_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 512,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split DataFrame into train/val/test and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, cate_dims, num_count
    """
    np.random.seed(seed)
    N = len(df)
    indices = np.random.permutation(N)

    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)
    n_train = N - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_ds = UpliftDataset(df.iloc[train_idx].reset_index(drop=True))
    val_ds = UpliftDataset(df.iloc[val_idx].reset_index(drop=True))
    test_ds = UpliftDataset(df.iloc[test_idx].reset_index(drop=True))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    cate_dims = get_cate_dims(df)
    _, num_cols = _identify_columns(df)
    num_count = len(num_cols)

    return train_loader, val_loader, test_loader, cate_dims, num_count

def get_dataloaders_from_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 512,
):
    """
    Return DataLoaders from pre-split DataFrames (e.g. Hillstrom datasets).
    """
    train_ds = UpliftDataset(train_df)
    val_ds = UpliftDataset(val_df)
    test_ds = UpliftDataset(test_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    cate_dims = get_cate_dims(train_df)
    _, num_cols = _identify_columns(train_df)
    num_count = len(num_cols)

    return train_loader, val_loader, test_loader, cate_dims, num_count


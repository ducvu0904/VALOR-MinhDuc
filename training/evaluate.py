"""
Evaluation utilities for VALOR experiments.

Computes the 5 metrics from §5.3:
  1. AUUC
  2. Qini (AUQC)
  3. Lift@30
  4. KRCC
  5. Inference latency
"""

import time
import torch
import numpy as np
import pandas as pd

from utils.metrics import auuc, auqc, lift, krcc


def evaluate_dnn_model(model, test_loader, device="cpu"):
    """
    Evaluate a DNN uplift model on the test set.

    Parameters
    ----------
    model       : nn.Module with ``predict_uplift(x_cat, x_num)`` method
    test_loader : DataLoader
    device      : str

    Returns
    -------
    dict with keys: auuc, qini, lift_30, krcc, latency_ms
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_treatments = []
    all_true_tau = []

    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            x_cat, x_num, treatment, label, true_tau = [
                b.to(device) for b in batch
            ]

            start = time.perf_counter()
            uplift_pred = model.predict_uplift(x_cat, x_num)
            elapsed = time.perf_counter() - start

            total_time += elapsed
            total_samples += len(label)

            all_preds.append(uplift_pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_treatments.append(treatment.cpu().numpy())
            all_true_tau.append(true_tau.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    treatments = np.concatenate(all_treatments)
    true_taus = np.concatenate(all_true_tau)

    # Compute metrics
    auuc_score = auuc(labels, treatments, preds, plot=False)
    qini_score = auqc(labels, treatments, preds, plot=False)
    lift_30 = lift(labels, treatments, preds, h=0.3)
    krcc_score = krcc(labels, treatments, preds)

    latency_ms = (total_time / max(total_samples, 1)) * 1000  # per sample

    return {
        "auuc": auuc_score,
        "qini": qini_score,
        "lift_30": lift_30,
        "krcc": krcc_score,
        "latency_ms": latency_ms,
    }


def evaluate_tree_model(model, X_test, treatment_test, y_test, true_tau_test=None):
    """
    Evaluate a tree-based model (CausalForest, ZILN-GBDT).

    Parameters
    ----------
    model          : object with ``predict_uplift(X)`` method
    X_test         : (N, d) ndarray
    treatment_test : (N,)
    y_test         : (N,)
    true_tau_test  : (N,) or None

    Returns
    -------
    dict with keys: auuc, qini, lift_30, krcc, latency_ms
    """
    start = time.perf_counter()
    preds = model.predict_uplift(X_test)
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / max(len(X_test), 1)) * 1000

    auuc_score = auuc(y_test, treatment_test, preds, plot=False)
    qini_score = auqc(y_test, treatment_test, preds, plot=False)
    lift_30 = lift(y_test, treatment_test, preds, h=0.3)
    krcc_score = krcc(y_test, treatment_test, preds)

    return {
        "auuc": auuc_score,
        "qini": qini_score,
        "lift_30": lift_30,
        "krcc": krcc_score,
        "latency_ms": latency_ms,
    }


def format_results_table(results: dict, title: str = "Results"):
    """
    Format a dict of {model_name: metrics_dict} as a pandas DataFrame
    matching Table 1 layout.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "AUUC": f"{metrics['auuc']:.4f}",
            "Qini": f"{metrics['qini']:.4f}",
            "Lift@30": f"{metrics['lift_30']:.2f}" if metrics['lift_30'] is not None and not np.isnan(metrics['lift_30']) else "N/A",
            "KRCC": f"{metrics['krcc']:.4f}" if metrics['krcc'] is not None and not np.isnan(metrics['krcc']) else "N/A",
            "Latency (ms)": f"{metrics['latency_ms']:.4f}",
        })
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}\n")
    return df

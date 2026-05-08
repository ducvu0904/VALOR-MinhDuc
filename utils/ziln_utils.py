"""
Shared ZILN (Zero-Inflated LogNormal) utility functions.

Used by both DNN heads (srm_heads.py, valor_net.py) and the GBDT variant
(ziln_gbdt.py) to ensure a single, consistent implementation of the
E[y] = π · exp(μ + σ²/2) formula referenced throughout the paper.
"""

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIGMA_MIN = 0.1
SIGMA_MAX = 4.0


# ---------------------------------------------------------------------------
# Torch helpers (used by DNN path)
# ---------------------------------------------------------------------------

def clamp_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """Clamp σ to [SIGMA_MIN, SIGMA_MAX] for numerical stability."""
    return sigma.clamp(min=SIGMA_MIN, max=SIGMA_MAX)


def ziln_expected_value(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute E[Y] under the Zero-Inflated LogNormal mixture.

        E[Y] = π · exp(μ + σ²/2)

    Parameters
    ----------
    pi : Tensor
        Hurdle / conversion probability ∈ (0, 1).  
        **Must already be passed through sigmoid** (not a raw logit).
    mu : Tensor
        Mean of log-revenue.
    sigma : Tensor
        Std-dev of log-revenue (will be clamped).

    Returns
    -------
    Tensor  — expected revenue, same shape as inputs.
    """
    sigma = clamp_sigma(sigma)
    return pi * torch.exp(mu + sigma.pow(2) / 2.0)


def ziln_uplift(
    pi_1: torch.Tensor, mu_1: torch.Tensor, sigma_1: torch.Tensor,
    pi_0: torch.Tensor, mu_0: torch.Tensor, sigma_0: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CATE (uplift) under ZILN:
        τ̂(x) = E[Y|T=1] − E[Y|T=0]
    """
    return (
        ziln_expected_value(pi_1, mu_1, sigma_1)
        - ziln_expected_value(pi_0, mu_0, sigma_0)
    )


# ---------------------------------------------------------------------------
# NumPy helpers (used by GBDT path)
# ---------------------------------------------------------------------------

def clamp_sigma_np(sigma: np.ndarray) -> np.ndarray:
    """Clamp σ (NumPy) to [SIGMA_MIN, SIGMA_MAX]."""
    return np.clip(sigma, SIGMA_MIN, SIGMA_MAX)


def ziln_expected_value_np(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """NumPy variant of :func:`ziln_expected_value`."""
    sigma = clamp_sigma_np(sigma)
    return pi * np.exp(mu + sigma ** 2 / 2.0)


def ziln_uplift_np(
    pi_1: np.ndarray, mu_1: np.ndarray, sigma_1: np.ndarray,
    pi_0: np.ndarray, mu_0: np.ndarray, sigma_0: np.ndarray,
) -> np.ndarray:
    """NumPy variant of :func:`ziln_uplift`."""
    return (
        ziln_expected_value_np(pi_1, mu_1, sigma_1)
        - ziln_expected_value_np(pi_0, mu_0, sigma_0)
    )

"""
Sparse-Revenue Mixture (SRM) Heads  (§4.1 of the paper).

Each treatment branch outputs three components:
    π  — hurdle / conversion probability (logit)
    μ  — conditional log-revenue mean
    σ  — log-revenue std, passed through Softplus + clamp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ziln_utils import SIGMA_MIN, SIGMA_MAX


class SRMHead(nn.Module):
    """
    Single SRM head for one treatment branch.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the shared representation feeding this head.
    hidden_dim : int
        Width of the hidden layer inside each sub-head.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 100):
        super().__init__()
        # π head  (hurdle probability)
        self.pi_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        # μ head  (conditional log-revenue mean)
        self.mu_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        # σ head  (log-revenue std — Softplus + clamp)
        self.sigma_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor):
        """
        Parameters
        ----------
        z : (B, input_dim)

        Returns
        -------
        pi_logit : (B,)  raw logit — apply sigmoid externally when needed
        mu       : (B,)
        sigma    : (B,)  already positive via Softplus, then clamped
        """
        pi_logit = self.pi_head(z).squeeze(-1)
        mu = self.mu_head(z).squeeze(-1)
        sigma = self.sigma_head(z).squeeze(-1)
        sigma = sigma.clamp(min=SIGMA_MIN, max=SIGMA_MAX)
        return pi_logit, mu, sigma

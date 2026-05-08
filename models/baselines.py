"""Baseline Models for Uplift Modeling.

Each DNN baseline supports a ``use_ziln`` flag:
  - ``use_ziln=False`` → scalar output per branch (standard MSE uplift)
  - ``use_ziln=True``  → 3-output head (π logit, μ, σ) per branch for ZILN

Baselines
---------
  - TARNet              — shared-bottom, two outcome heads
  - DragonNet           — TARNet + propensity head
  - CFR (WASS / MMD)    — TARNet + IPM regulariser on representations
  - UniTE               — Robinson decomposition (prognostic + treatment)
  - EUEN                — Explicit uplift estimation with bias correction
  - TLearner            — two independent networks
  - SLearner            — single network, treatment concatenated
  - CausalForestWrapper — econml CausalForestDML
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.ziln_utils import ziln_expected_value


# =====================================================================
#  Utility: flexible output head (scalar or ZILN triple)
# =====================================================================
class _OutcomeHead(nn.Module):
    """A single outcome head that outputs either 1 scalar or 3 ZILN params."""

    def __init__(self, in_dim: int, hidden_dim: int = 100, use_ziln: bool = False):
        super().__init__()
        self.use_ziln = use_ziln

        if use_ziln:
            # 3 sub-heads: π (logit), μ, σ (via Softplus)
            self.pi_head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.mu_head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.sigma_head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, 1), nn.Softplus(),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, z):
        if self.use_ziln:
            pi = self.pi_head(z).squeeze(-1)
            mu = self.mu_head(z).squeeze(-1)
            sigma = self.sigma_head(z).squeeze(-1).clamp(min=0.1, max=4.0)
            return pi, mu, sigma
        else:
            return self.head(z).squeeze(-1)


# =====================================================================
#  Shared embedding layer
# =====================================================================
class _EmbeddingBlock(nn.Module):
    """Embed categoricals + concat numericals into a single vector."""

    def __init__(self, cate_dims, num_count, emb_size=10):
        super().__init__()
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(dim, emb_size) for dim in cate_dims]
        )
        self.num_count = num_count
        self.output_dim = len(cate_dims) * emb_size + num_count

    def forward(self, x_cat, x_num):
        parts = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embeds)]
        if self.num_count > 0:
            parts.append(x_num.float())
        return torch.cat(parts, dim=1)


# =====================================================================
# 1.  TARNet  (Treatment-Agnostic Representation Network)
# =====================================================================
class TARNet(nn.Module):
    """
    Shared-bottom + two treatment-specific outcome heads.

    Parameters
    ----------
    cate_dims     : list[int]  vocabulary sizes for each categorical feature
    num_count     : int        number of numerical features
    shared_hidden : int        width of shared representation layers
    outcome_hidden: int        width of outcome head layers
    use_ziln      : bool       if True, outcome heads output (π, μ, σ)
    """

    def __init__(
        self,
        cate_dims,
        num_count,
        shared_hidden: int = 200,
        outcome_hidden: int = 100,
        use_ziln: bool = False,
    ):
        super().__init__()
        self.use_ziln = use_ziln

        self.embedding = _EmbeddingBlock(cate_dims, num_count)
        in_dim = self.embedding.output_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
        )
        self.shared_out_dim = shared_hidden

        self.y0_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)
        self.y1_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)

    def get_representation(self, x_cat, x_num):
        """Return shared representation (before outcome heads)."""
        emb = self.embedding(x_cat, x_num)
        return self.shared(emb)

    def forward(self, x_cat, x_num):
        z = self.get_representation(x_cat, x_num)
        y0 = self.y0_head(z)
        y1 = self.y1_head(z)
        return y0, y1

    def predict_uplift(self, x_cat, x_num):
        """Return scalar uplift per sample."""
        y0, y1 = self.forward(x_cat, x_num)
        if self.use_ziln:
            pi0, mu0, sig0 = y0
            pi1, mu1, sig1 = y1
            ev0 = ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
            ev1 = ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
            return ev1 - ev0
        else:
            return y1 - y0


# =====================================================================
# 2.  DragonNet  (TARNet + propensity head)
# =====================================================================
class DragonNet(nn.Module):
    """
    DragonNet adds a propensity-score head to TARNet for
    targeted regularisation.
    """

    def __init__(
        self,
        cate_dims,
        num_count,
        shared_hidden: int = 200,
        outcome_hidden: int = 100,
        use_ziln: bool = False,
    ):
        super().__init__()
        self.use_ziln = use_ziln

        self.embedding = _EmbeddingBlock(cate_dims, num_count)
        in_dim = self.embedding.output_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
        )
        self.shared_out_dim = shared_hidden

        self.y0_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)
        self.y1_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)

        # Propensity head  P(T=1 | X)
        self.propensity_head = nn.Sequential(
            nn.Linear(shared_hidden, outcome_hidden), nn.ELU(),
            nn.Linear(outcome_hidden, 1),
        )

    def get_representation(self, x_cat, x_num):
        emb = self.embedding(x_cat, x_num)
        return self.shared(emb)

    def forward(self, x_cat, x_num):
        z = self.get_representation(x_cat, x_num)
        y0 = self.y0_head(z)
        y1 = self.y1_head(z)
        propensity_logit = self.propensity_head(z).squeeze(-1)
        return y0, y1, propensity_logit

    def predict_uplift(self, x_cat, x_num):
        out = self.forward(x_cat, x_num)
        y0, y1 = out[0], out[1]
        if self.use_ziln:
            pi0, mu0, sig0 = y0
            pi1, mu1, sig1 = y1
            ev0 = ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
            ev1 = ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
            return ev1 - ev0
        else:
            return y1 - y0


# =====================================================================
# 3.  CFR  (Counterfactual Regression — IPM regularisation)
# =====================================================================

def _gaussian_kernel(x, y, sigma=1.0):
    """Gaussian kernel for MMD."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    return torch.exp(-((tiled_x - tiled_y).pow(2).sum(2)) / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    """Maximum Mean Discrepancy with Gaussian kernel."""
    k_xx = _gaussian_kernel(x, x, sigma)
    k_yy = _gaussian_kernel(y, y, sigma)
    k_xy = _gaussian_kernel(x, y, sigma)
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()


def compute_wasserstein(x, y, p=1, n_iter=100, reg=0.1):
    """
    Sinkhorn approximation of the Wasserstein-p distance.
    Differentiable proxy used for CFR-WASS.
    """
    n = x.size(0)
    m = y.size(0)
    if n == 0 or m == 0:
        return torch.tensor(0.0, device=x.device)

    # Cost matrix  C_ij = ||x_i − y_j||^p
    C = torch.cdist(x, y, p=2).pow(p)

    # Sinkhorn iterations
    K = torch.exp(-C / reg)
    u = torch.ones(n, 1, device=x.device) / n
    for _ in range(n_iter):
        v = 1.0 / (K.t() @ u + 1e-8) / m
        u = 1.0 / (K @ v + 1e-8) / n

    T = u * K * v.t()
    return (T * C).sum()


class CFR(nn.Module):
    """
    Counterfactual Regression: TARNet + IPM regulariser.

    Parameters
    ----------
    mode : str   'mmd' or 'wass'
    lambda_ipm : float   weight for IPM penalty
    """

    def __init__(
        self,
        cate_dims,
        num_count,
        shared_hidden: int = 200,
        outcome_hidden: int = 100,
        use_ziln: bool = False,
        mode: str = "wass",
        lambda_ipm: float = 1.0,
    ):
        super().__init__()
        assert mode in ("mmd", "wass"), f"Unknown IPM mode: {mode}"
        self.mode = mode
        self.lambda_ipm = lambda_ipm
        self.use_ziln = use_ziln

        self.embedding = _EmbeddingBlock(cate_dims, num_count)
        in_dim = self.embedding.output_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
        )
        self.shared_out_dim = shared_hidden

        self.y0_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)
        self.y1_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)

    def get_representation(self, x_cat, x_num):
        emb = self.embedding(x_cat, x_num)
        return self.shared(emb)

    def compute_ipm(self, z, t):
        """Compute the IPM penalty between treated / control representations."""
        t_flat = t.view(-1)
        z_t = z[t_flat == 1]
        z_c = z[t_flat == 0]
        if z_t.size(0) == 0 or z_c.size(0) == 0:
            return torch.tensor(0.0, device=z.device)
        if self.mode == "mmd":
            return compute_mmd(z_t, z_c)
        else:
            return compute_wasserstein(z_t, z_c)

    def forward(self, x_cat, x_num, treatment=None):
        z = self.get_representation(x_cat, x_num)
        y0 = self.y0_head(z)
        y1 = self.y1_head(z)

        ipm_loss = None
        if treatment is not None:
            ipm_loss = self.compute_ipm(z, treatment)

        return y0, y1, ipm_loss

    def predict_uplift(self, x_cat, x_num):
        y0, y1, _ = self.forward(x_cat, x_num)
        if self.use_ziln:
            pi0, mu0, sig0 = y0
            pi1, mu1, sig1 = y1
            ev0 = ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
            ev1 = ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
            return ev1 - ev0
        else:
            return y1 - y0


# =====================================================================
# 4.  UniTE  (Unified Treatment Effect — Robinson Decomposition)
# =====================================================================
class UniTE(nn.Module):
    """
    Y = μ(x) + T·τ(x) + ε

    Decomposes into:
     - Prognostic function μ(x) — baseline outcome
     - Treatment effect τ(x) — heterogeneous uplift
    """

    def __init__(
        self,
        cate_dims,
        num_count,
        shared_hidden: int = 200,
        outcome_hidden: int = 100,
        use_ziln: bool = False,
    ):
        super().__init__()
        self.use_ziln = use_ziln

        self.embedding = _EmbeddingBlock(cate_dims, num_count)
        in_dim = self.embedding.output_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
        )
        self.shared_out_dim = shared_hidden

        # Prognostic head μ(x)
        self.mu_prog_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)
        # Treatment effect head τ(x)
        self.tau_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)

    def get_representation(self, x_cat, x_num):
        emb = self.embedding(x_cat, x_num)
        return self.shared(emb)

    def forward(self, x_cat, x_num):
        z = self.get_representation(x_cat, x_num)
        mu_prog = self.mu_prog_head(z)
        tau = self.tau_head(z)
        return mu_prog, tau

    def predict_outcome(self, x_cat, x_num, treatment):
        """Predict Y = μ(x) + T·τ(x)."""
        mu_prog, tau = self.forward(x_cat, x_num)
        if self.use_ziln:
            # For ZILN mode, return the two heads as (y0, y1) equivalent
            # y0 = mu_prog, y1 = mu_prog + tau in expected-value space
            return mu_prog, tau
        else:
            t = treatment.float()
            return mu_prog + t * tau

    def predict_uplift(self, x_cat, x_num):
        _, tau = self.forward(x_cat, x_num)
        if self.use_ziln:
            pi, mu, sig = tau
            # For UniTE the tau head directly gives the treatment effect
            # Using E[Y] = π·exp(μ + σ²/2) as the uplift magnitude
            return ziln_expected_value(torch.sigmoid(pi), mu, sig)
        else:
            return tau


# =====================================================================
# 5.  EUEN  (Explicit Uplift Estimation Network)
# =====================================================================
class EUEN(nn.Module):
    """
    Explicit Uplift Estimation Network.

    Architecture: control head + explicit uplift head.
    Y_hat = control_head(x) + T * uplift_head(x)
    """

    def __init__(
        self,
        cate_dims,
        num_count,
        shared_hidden: int = 200,
        outcome_hidden: int = 100,
        use_ziln: bool = False,
    ):
        super().__init__()
        self.use_ziln = use_ziln

        self.embedding = _EmbeddingBlock(cate_dims, num_count)
        in_dim = self.embedding.output_dim

        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
            nn.Linear(shared_hidden, shared_hidden), nn.ELU(),
        )
        self.shared_out_dim = shared_hidden

        # Control head = E[Y | T=0, X]
        self.control_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)
        # Uplift head = τ(x)
        self.uplift_head = _OutcomeHead(shared_hidden, outcome_hidden, use_ziln)

    def get_representation(self, x_cat, x_num):
        emb = self.embedding(x_cat, x_num)
        return self.shared(emb)

    def forward(self, x_cat, x_num):
        z = self.get_representation(x_cat, x_num)
        control = self.control_head(z)
        uplift = self.uplift_head(z)
        return control, uplift

    def predict_uplift(self, x_cat, x_num):
        _, uplift = self.forward(x_cat, x_num)
        if self.use_ziln:
            pi, mu, sig = uplift
            return ziln_expected_value(torch.sigmoid(pi), mu, sig)
        else:
            return uplift


# =====================================================================
# 6.  T-Learner  (two independent networks)
# =====================================================================
class TLearner(nn.Module):
    """Two completely independent networks for T=0 and T=1."""

    def __init__(
        self,
        cate_dims,
        num_count,
        hidden: int = 200,
        outcome_hidden: int = 100,
        use_ziln: bool = False,
    ):
        super().__init__()
        self.use_ziln = use_ziln

        self.emb_0 = _EmbeddingBlock(cate_dims, num_count)
        self.emb_1 = _EmbeddingBlock(cate_dims, num_count)
        in_dim = self.emb_0.output_dim

        self.net_0 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.net_1 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )

        self.y0_head = _OutcomeHead(hidden, outcome_hidden, use_ziln)
        self.y1_head = _OutcomeHead(hidden, outcome_hidden, use_ziln)

    def forward(self, x_cat, x_num):
        e0 = self.emb_0(x_cat, x_num)
        e1 = self.emb_1(x_cat, x_num)
        z0 = self.net_0(e0)
        z1 = self.net_1(e1)
        y0 = self.y0_head(z0)
        y1 = self.y1_head(z1)
        return y0, y1

    def predict_uplift(self, x_cat, x_num):
        y0, y1 = self.forward(x_cat, x_num)
        if self.use_ziln:
            pi0, mu0, sig0 = y0
            pi1, mu1, sig1 = y1
            ev0 = ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
            ev1 = ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
            return ev1 - ev0
        else:
            return y1 - y0


# =====================================================================
# 7.  S-Learner  (single network, treatment concatenated)
# =====================================================================
class SLearner(nn.Module):
    """
    Single network — treatment indicator is appended as an extra feature.
    No ZILN variant (per plan).
    """

    def __init__(
        self,
        cate_dims,
        num_count,
        hidden: int = 200,
        outcome_hidden: int = 100,
    ):
        super().__init__()
        self.use_ziln = False

        self.embedding = _EmbeddingBlock(cate_dims, num_count)
        # +1 for treatment indicator
        in_dim = self.embedding.output_dim + 1

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, outcome_hidden), nn.ELU(),
            nn.Linear(outcome_hidden, 1),
        )

    def forward(self, x_cat, x_num, treatment):
        emb = self.embedding(x_cat, x_num)
        t = treatment.float().unsqueeze(-1)
        x = torch.cat([emb, t], dim=1)
        return self.net(x).squeeze(-1)

    def predict_uplift(self, x_cat, x_num):
        emb = self.embedding(x_cat, x_num)
        B = emb.size(0)
        dev = emb.device

        t0 = torch.zeros(B, 1, device=dev)
        t1 = torch.ones(B, 1, device=dev)

        y0 = self.net(torch.cat([emb, t0], dim=1)).squeeze(-1)
        y1 = self.net(torch.cat([emb, t1], dim=1)).squeeze(-1)
        return y1 - y0


# =====================================================================
# 8.  Causal Forest Wrapper  (econml)
# =====================================================================
class CausalForestWrapper:
    """
    Wrapper around ``econml.dml.CausalForestDML``.
    
    Not an nn.Module — uses sklearn-style fit / predict API.
    20 estimators, max_depth=6 as per paper.
    """

    def __init__(self, n_estimators: int = 20, max_depth: int = 6, random_state: int = 42):
        from econml.dml import CausalForestDML
        self.model = CausalForestDML(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X, treatment, y):
        """
        Parameters
        ----------
        X         : array (N, d)
        treatment : array (N,)
        y         : array (N,)
        """
        self.model.fit(y, T=treatment, X=X)
        return self

    def predict_uplift(self, X):
        """Return estimated CATE for each sample."""
        return self.model.effect(X).flatten()

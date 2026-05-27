"""EFIN — Explicit Feature Interaction-aware Uplift Network.

Architecture (per paper §4):
  1. Feature Encoder     — embeds non-treatment and treatment features separately
  2. Self-Interaction    — self-attention on non-treatment features → ŷ(0)
  3. Treatment-Aware     — attention between treatment and feature embeddings → τ̂(x)
  4. Intervention Const. — auxiliary classifier trained with inverted labels

Final prediction:  ŷ(k) = ŷ(0) + τ̂_k(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines import _EmbeddingBlock, _OutcomeHead
from utils.ziln_utils import ziln_expected_value


class EFIN(nn.Module):
    """
    Explicit Feature Interaction-aware Uplift Network.

    Parameters
    ----------
    cate_dims     : list[int]  vocabulary sizes for each categorical feature
    num_count     : int        number of numerical (non-treatment) features
    treatment_dim : int        number of treatment feature columns (1 for binary)
    hidden_dim    : int        hidden size throughout
    emb_size      : int        embedding size per categorical feature
    use_ziln      : bool       if True, outcome heads output (π_logit, μ, σ)
    """

    def __init__(
        self,
        cate_dims,
        num_count,
        treatment_dim: int = 1,
        hidden_dim: int = 128,
        emb_size: int = 10,
        use_ziln: bool = False,
    ):
        super().__init__()
        self.use_ziln = use_ziln
        self.hidden_dim = hidden_dim

        # ── 1. Feature Encoder ────────────────────────────────────────────────
        # Non-treatment features
        self.x_embedding = _EmbeddingBlock(cate_dims, num_count, emb_size)
        self.x_feat_dim = self.x_embedding.output_dim
        self.x_proj = nn.Linear(self.x_feat_dim, hidden_dim)

        # Treatment feature encoder (binary scalar → hidden_dim)
        self.treatment_encoder = nn.Linear(1, hidden_dim)

        # ── 2. Self-Interaction Module ────────────────────────────────────────
        # Equations (4)–(7): multi-head self-attention on e_x, then MLP → ŷ(0)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.self_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
        )
        self.y0_head = _OutcomeHead(hidden_dim // 2, hidden_dim // 2, use_ziln)

        # ── 3. Treatment-Aware Interaction Module ─────────────────────────────
        # Equations (8)–(12): treatment acts as query, non-treatment as K/V
        self.W_t0 = nn.Linear(hidden_dim, 1)          # scalar attention scorer
        self.W_t1 = nn.Linear(hidden_dim, hidden_dim)  # transform treatment
        self.W_t2 = nn.Linear(hidden_dim, hidden_dim)  # transform each x feature
        self.treat_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ELU(),
        )
        self.tau_head = _OutcomeHead(hidden_dim // 2, hidden_dim // 2, use_ziln)

        # ── 4. Intervention Constraint Module ─────────────────────────────────
        # Equations (13)–(14): linear classifier on e_xt with inverted labels
        self.intervention_head = nn.Linear(hidden_dim, 1)

        # Compatibility attribute for RERUMWrapper
        self.shared_out_dim = hidden_dim

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x_cat, x_num, treatment):
        """
        Parameters
        ----------
        x_cat     : (B, n_cat) long tensor
        x_num     : (B, num_count) float tensor
        treatment : (B,) float tensor  (0 or 1)

        Returns
        -------
        y0_out        : scalar or (pi, mu, sigma) — control prediction
        tau_out       : scalar or (pi, mu, sigma) — uplift prediction
        t_hat_logit   : (B,) — treatment group logit (for L_C)
        e_xt          : (B, hidden_dim) — treatment-interaction representation
        """
        # ── Encode non-treatment features ─────────────────────────────────────
        e_x = self.x_embedding(x_cat, x_num)            # (B, x_feat_dim)
        e_x_proj = F.relu(self.x_proj(e_x))             # (B, hidden_dim)

        # ── Encode treatment feature ──────────────────────────────────────────
        e_t = F.relu(self.treatment_encoder(
            treatment.float().unsqueeze(-1)              # (B, 1)
        ))                                               # (B, hidden_dim)

        # ── Self-Interaction (ŷ(0)) ───────────────────────────────────────────
        # Treat e_x_proj as a single-token sequence for multi-head attention
        e_x_seq = e_x_proj.unsqueeze(1)                 # (B, 1, hidden_dim)
        attn_out, _ = self.self_attn(e_x_seq, e_x_seq, e_x_seq)
        z_self = self.self_mlp(attn_out.squeeze(1))     # (B, hidden_dim//2)
        y0_out = self.y0_head(z_self)

        # ── Treatment-Aware Interaction (τ̂(x)) ───────────────────────────────
        # Gate: alpha = sigmoid(W_t0(relu(W_t1(e_t) + W_t2(e_x_proj))))
        alpha = torch.sigmoid(
            self.W_t0(F.relu(self.W_t1(e_t) + self.W_t2(e_x_proj)))
        )                                                # (B, 1)
        e_xt = alpha * e_x_proj                         # (B, hidden_dim)
        z_treat = self.treat_mlp(e_xt)                  # (B, hidden_dim//2)
        tau_out = self.tau_head(z_treat)

        # ── Intervention Constraint ───────────────────────────────────────────
        t_hat_logit = self.intervention_head(e_xt).squeeze(-1)  # (B,)

        return y0_out, tau_out, t_hat_logit, e_xt

    # ── Inference helpers ─────────────────────────────────────────────────────

    def predict_uplift(self, x_cat, x_num):
        """
        Return scalar ITE estimate per sample.

        Passes treatment=1 (binary treated), returns τ̂(x) only —
        consistent with §4.2.5 of the paper (inference uses TAI module).
        """
        B = x_cat.size(0)
        device = x_cat.device
        treatment = torch.ones(B, device=device)

        _, tau_out, _, _ = self.forward(x_cat, x_num, treatment)

        if self.use_ziln:
            pi, mu, sigma = tau_out
            return ziln_expected_value(torch.sigmoid(pi), mu, sigma)
        else:
            return tau_out

    def get_representation(self, x_cat, x_num):
        """Return the projected non-treatment representation (for RERUM compatibility)."""
        e_x = self.x_embedding(x_cat, x_num)
        return F.relu(self.x_proj(e_x))

"""
RERUM Network — Rankability-Enhanced Revenue Uplift Modeling.

Usage::

    backbone = TARNet(cate_dims, num_count)   # use_ziln=False is fine
    model    = RERUMWrapper(backbone)
    # -> RERUM-TARNet

The wrapper:
  1. Extracts the shared representation from the backbone.
  2. Routes it through two independent ZILNHead branches (control / treatment).
  3. Computes expected revenues ŷ₀, ŷ₁ and uplift τ̂ = ŷ₁ − ŷ₀.

Any backbone that exposes:
  - ``get_representation(x_cat, x_num) → z``
  - ``shared_out_dim``  (int)
is compatible.
"""

import torch
import torch.nn as nn

from models.srm_heads import SRMHead
from utils.ziln_utils import ziln_expected_value


class RERUMWrapper(nn.Module):
    """
    RERUM wrapper over any shared-representation backbone.

    Parameters
    ----------
    backbone     : nn.Module
        Baseline model with ``get_representation`` and ``shared_out_dim``.
    outcome_hidden : int
        Hidden width inside each ZILNHead branch.
    """

    def __init__(self, backbone: nn.Module, outcome_hidden: int = 128):
        super().__init__()
        assert hasattr(backbone, "get_representation"), (
            "RERUM requires backbone to expose get_representation(x_cat, x_num)."
        )
        assert hasattr(backbone, "shared_out_dim"), (
            "RERUM requires backbone to expose shared_out_dim (int)."
        )

        self.backbone = backbone
        repr_dim = backbone.shared_out_dim

        # Two ZILN heads — one per treatment arm
        # SRMHead is identical to ZILNHead in this repo (π, μ, σ per branch)
        self.ziln_y0 = SRMHead(repr_dim, outcome_hidden)
        self.ziln_y1 = SRMHead(repr_dim, outcome_hidden)

    # ------------------------------------------------------------------
    def forward(self, x_cat, x_num):
        """
        Parameters
        ----------
        x_cat : (B, n_cat)  LongTensor
        x_num : (B, n_num)  FloatTensor

        Returns
        -------
        y0_params : tuple (pi_logit, mu, sigma)  — control branch
        y1_params : tuple (pi_logit, mu, sigma)  — treatment branch
        extras    : dict — may include ipm_loss (CFR backbone)
        """
        z = self.backbone.get_representation(x_cat, x_num)

        y0_params = self.ziln_y0(z)   # (pi0, mu0, sig0)
        y1_params = self.ziln_y1(z)   # (pi1, mu1, sig1)

        extras = {}
        # CFR → IPM loss (already computed inside backbone.forward; we
        # re-compute from the representation so gradients flow)
        if hasattr(self.backbone, "compute_ipm"):
            # treatment is not available in this path — handled in trainer
            pass

        return y0_params, y1_params, extras

    # ------------------------------------------------------------------
    def forward_with_treatment(self, x_cat, x_num, treatment):
        """
        Extended forward used by the trainer to access IPM loss (CFR).

        Returns
        -------
        y0_params, y1_params, extras
          extras["ipm_loss"]   — IPM scalar if backbone is CFR
          extras["lambda_ipm"] — corresponding weight
        """
        z = self.backbone.get_representation(x_cat, x_num)

        y0_params = self.ziln_y0(z)
        y1_params = self.ziln_y1(z)

        extras = {}
        if hasattr(self.backbone, "compute_ipm"):
            extras["ipm_loss"] = self.backbone.compute_ipm(z, treatment)
            extras["lambda_ipm"] = getattr(self.backbone, "lambda_ipm", 1.0)

        return y0_params, y1_params, extras

    # ------------------------------------------------------------------
    def predict_uplift(self, x_cat, x_num):
        """
        Scalar uplift prediction τ̂(x) = E[Y|T=1] − E[Y|T=0].
        """
        z = self.backbone.get_representation(x_cat, x_num)

        pi0, mu0, sig0 = self.ziln_y0(z)
        pi1, mu1, sig1 = self.ziln_y1(z)

        ev0 = ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
        ev1 = ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
        return ev1 - ev0

"""
VALOR Network  — assembles backbone + gating + SRM heads + losses.

Usage::

    backbone = TARNet(cate_dims, num_count, use_ziln=True)
    model    = VALOR(backbone, use_gating=True)
    # → This gives VALOR-TARNet (one row in Table 1)

    backbone = CFR(cate_dims, num_count, use_ziln=True, mode='wass')
    model    = VALOR(backbone, use_gating=True)
    # → VALOR-CFR-WASS

The same ``VALOR`` class works with any backbone that exposes:
    - ``get_representation(x_cat, x_num) → z``
    - ``shared_out_dim``  (int)
    - ``use_ziln == True``
"""

import torch
import torch.nn as nn

from models.valor.gating import TreatmentGatedInteraction
from models.srm_heads import SRMHead
from utils.ziln_utils import ziln_expected_value


class VALOR(nn.Module):
    """
    VALOR wrapper that takes an existing backbone and augments it with:
      - (optional) Treatment-Gated Interaction (§4.3)
      - SRM heads for control & treatment branches (§4.1)

    Parameters
    ----------
    backbone     : nn.Module
        A baseline model (TARNet, DragonNet, CFR, …) with ``use_ziln=True``.
    use_gating   : bool
        Whether to apply Treatment-Gated Interaction.
    outcome_hidden : int
        Hidden width inside each SRM head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        use_gating: bool = True,
        outcome_hidden: int = 100,
    ):
        super().__init__()
        assert getattr(backbone, "use_ziln", False), (
            "VALOR requires the backbone to have use_ziln=True"
        )
        self.backbone = backbone
        self.use_gating = use_gating

        repr_dim = backbone.shared_out_dim

        # Optional gating module
        if use_gating:
            self.gating = TreatmentGatedInteraction(repr_dim, repr_dim)
        else:
            self.gating = None

        # SRM heads (replace backbone heads)
        self.srm_y0 = SRMHead(repr_dim, outcome_hidden)
        self.srm_y1 = SRMHead(repr_dim, outcome_hidden)

    # ------------------------------------------------------------------
    def _get_representation(self, x_cat, x_num, treatment=None):
        """
        Get shared representation from backbone, optionally gated.
        """
        z = self.backbone.get_representation(x_cat, x_num)

        if self.use_gating and treatment is not None:
            z = self.gating(z, treatment)

        return z

    # ------------------------------------------------------------------
    def forward(self, x_cat, x_num, treatment):
        """
        Parameters
        ----------
        x_cat     : (B, n_cat)  LongTensor
        x_num     : (B, n_num)  FloatTensor
        treatment : (B,)        0/1 FloatTensor

        Returns
        -------
        y0_params : tuple (pi_logit, mu, sigma)   for control
        y1_params : tuple (pi_logit, mu, sigma)   for treatment
        extras    : dict — may include propensity_logit, ipm_loss
        """
        z = self._get_representation(x_cat, x_num, treatment)

        y0_params = self.srm_y0(z)  # (pi_logit, mu, sigma)
        y1_params = self.srm_y1(z)

        extras = {}

        # DragonNet → propensity head
        if hasattr(self.backbone, "propensity_head"):
            extras["propensity_logit"] = self.backbone.propensity_head(z).squeeze(-1)

        # CFR → IPM loss
        if hasattr(self.backbone, "compute_ipm"):
            extras["ipm_loss"] = self.backbone.compute_ipm(z, treatment)
            extras["lambda_ipm"] = getattr(self.backbone, "lambda_ipm", 1.0)

        return y0_params, y1_params, extras

    # ------------------------------------------------------------------
    def predict_uplift(self, x_cat, x_num):
        """
        Predict scalar uplift τ̂(x) = E[Y|T=1] − E[Y|T=0].

        We need representations under *both* treatment conditions.
        If gating is enabled, we pass the representation through the gate
        twice (once per treatment arm) so the gate can modulate differently.
        """
        z_base = self.backbone.get_representation(x_cat, x_num)
        B = z_base.size(0)
        dev = z_base.device

        if self.use_gating:
            t0 = torch.zeros(B, device=dev)
            t1 = torch.ones(B, device=dev)
            z0 = self.gating(z_base, t0)
            z1 = self.gating(z_base, t1)
        else:
            z0 = z1 = z_base

        pi0, mu0, sig0 = self.srm_y0(z0)
        pi1, mu1, sig1 = self.srm_y1(z1)

        ev0 = ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
        ev1 = ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
        return ev1 - ev0

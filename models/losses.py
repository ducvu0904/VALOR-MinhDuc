"""
VALOR Loss Functions (§4.2 of the paper).

This module has **zero model dependencies** — every loss operates on raw
tensors and can be unit-tested on dummy data before any model exists.

Loss hierarchy
--------------
1. ZILNLoss            — standard ZILN negative log-likelihood
2. FocalZILNLoss       — adds focal modulation to the propensity BCE  (§4.2.1)
3. ValueWeightedRankingLoss — pairwise ranking weighted by value     (§4.2.2)
4. VALORLoss           — combines Focal-ZILN + ranking (+ optional IPM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ziln_utils import clamp_sigma, SIGMA_MIN, SIGMA_MAX

# Precomputed constant — avoids torch.tensor allocation every batch
LOG_2PI = math.log(2 * math.pi)


# ===================================================================
# 1.  Standard ZILN negative log-likelihood
# ===================================================================
class ZILNLoss(nn.Module):
    """
    Standard Zero-Inflated LogNormal loss.

    L = L_prop + L_rev

    • L_prop = BCE(π_logit, 1[y>0])        — propensity component
    • L_rev  = −logN(log(y); μ, σ²)         — revenue component (y>0 only)
    """

    def forward(self, pi_logit, mu, sigma, y):
        """
        Parameters
        ----------
        pi_logit : (B,)  raw logit for hurdle probability
        mu       : (B,)  predicted log-revenue mean
        sigma    : (B,)  predicted log-revenue std (pre-clamp)
        y        : (B,)  observed outcome (≥ 0)

        Returns
        -------
        scalar loss
        """
        sigma = clamp_sigma(sigma)
        is_positive = (y > 0).float()

        # Propensity loss (full batch)
        loss_prop = F.binary_cross_entropy_with_logits(
            pi_logit, is_positive, reduction="mean"
        )

        # Revenue loss (positive-only)
        pos_mask = y > 0
        if pos_mask.sum() > 0:
            log_y = torch.log(y[pos_mask])
            mu_pos = mu[pos_mask]
            sigma_pos = sigma[pos_mask]
            # NLL of LogNormal
            loss_rev = (
                torch.log(sigma_pos)
                + 0.5 * LOG_2PI
                + (log_y - mu_pos).pow(2) / (2 * sigma_pos.pow(2))
            ).mean()
        else:
            loss_rev = torch.tensor(0.0, device=y.device)

        return loss_prop + loss_rev


# ===================================================================
# 2.  Focal-ZILN Loss  (§4.2.1)
# ===================================================================
class FocalZILNLoss(nn.Module):
    """
    Focal-ZILN = Focal propensity + standard LogNormal revenue.

    L_focal_prop = −α (1 − p_t)^γ · log(p_t)

    γ (gamma) : focusing parameter — down-weights easy negatives
    α (alpha) : balance factor
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pi_logit, mu, sigma, y):
        sigma = clamp_sigma(sigma)
        is_positive = (y > 0).float()

        # ---------- Focal propensity loss ----------
        p = torch.sigmoid(pi_logit)
        # p_t = probability assigned to the TRUE class
        p_t = p * is_positive + (1 - p) * (1 - is_positive)
        # focal weight
        focal_weight = (1 - p_t).pow(self.gamma)
        # alpha weighting (α for positives, 1−α for negatives)
        alpha_t = self.alpha * is_positive + (1 - self.alpha) * (1 - is_positive)
        # Cross-entropy per sample
        bce = F.binary_cross_entropy_with_logits(
            pi_logit, is_positive, reduction="none"
        )
        loss_prop = (alpha_t * focal_weight * bce).mean()

        # ---------- Revenue loss ----------
        pos_mask = y > 0
        if pos_mask.sum() > 0:
            log_y = torch.log(y[pos_mask])
            mu_pos = mu[pos_mask]
            sigma_pos = sigma[pos_mask]
            loss_rev = (
                torch.log(sigma_pos)
                + 0.5 * LOG_2PI
                + (log_y - mu_pos).pow(2) / (2 * sigma_pos.pow(2))
            ).mean()
        else:
            loss_rev = torch.tensor(0.0, device=y.device)

        return loss_prop + loss_rev


# ===================================================================
# 3.  Value-Weighted Ranking Loss  (§4.2.2)
# ===================================================================
class ValueWeightedRankingLoss(nn.Module):
    """
    Pairwise ranking loss weighted by financial magnitude.

    L_V-Rank = Σ_{i,j} w_ij · log(1 + exp(−sign(z_i − z_j) · (τ̂_i − τ̂_j)))

    where  w_ij = log(1 + |z_i − z_j|)

    Fully vectorised O(n²) — at batch=512 this is 262 K pairs, fine on GPU.
    """

    def forward(self, tau_hat, z):
        """
        Parameters
        ----------
        tau_hat : (B,)  predicted uplift scores
        z       : (B,)  transformed ground-truth outcome
                        (used as proxy for true uplift ranking)

        Returns
        -------
        scalar loss
        """
        # Pairwise differences via broadcasting
        z_diff = z.unsqueeze(1) - z.unsqueeze(0)          # (B, B)
        tau_diff = tau_hat.unsqueeze(1) - tau_hat.unsqueeze(0)  # (B, B)

        # sign matrix
        sign = torch.sign(z_diff)

        # Pair weight  w_ij = log(1 + |z_i − z_j|)
        w = torch.log1p(z_diff.abs())

        # Logistic surrogate
        logistic = torch.log1p(torch.exp(-sign * tau_diff))

        # Weighted sum (exclude diagonal where z_diff == 0)
        mask = (z_diff.abs() > 1e-8).float()
        loss = (w * logistic * mask).sum() / mask.sum().clamp(min=1.0)

        return loss


# ===================================================================
# 4.  Combined VALOR Loss
# ===================================================================
class VALORLoss(nn.Module):
    """
    Total VALOR loss:
        L = L_FL-ZILN + λ_rank · L_V-Rank  [+ λ_ipm · L_IPM (external)]

    Parameters
    ----------
    gamma, alpha : focal hyper-params
    lambda_rank  : weight for the ranking loss (default 1.0)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        lambda_rank: float = 1.0,
    ):
        super().__init__()
        self.focal_ziln = FocalZILNLoss(gamma=gamma, alpha=alpha)
        self.ranking = ValueWeightedRankingLoss()
        self.lambda_rank = lambda_rank

    def forward(
        self,
        pi_logit, mu, sigma, y,
        tau_hat, z,
        ipm_loss=None,
        lambda_ipm=0.0,
    ):
        """
        Parameters
        ----------
        pi_logit, mu, sigma, y : same as FocalZILNLoss
        tau_hat : (B,)  predicted CATE for ranking
        z       : (B,)  ground-truth proxy for ranking
        ipm_loss : optional pre-computed IPM scalar (e.g. from CFR backbone)
        lambda_ipm : weight for IPM regularizer

        Returns
        -------
        total_loss, dict of component losses
        """
        l_fl_ziln = self.focal_ziln(pi_logit, mu, sigma, y)
        l_rank = self.ranking(tau_hat, z)

        total = l_fl_ziln + self.lambda_rank * l_rank

        if ipm_loss is not None and lambda_ipm > 0:
            total = total + lambda_ipm * ipm_loss

        components = {
            "focal_ziln": l_fl_ziln.item(),
            "ranking": l_rank.item(),
        }
        if ipm_loss is not None:
            components["ipm"] = ipm_loss.item()

        return total, components


# ===================================================================
# 5.  RERUM Ranking Losses
# ===================================================================
class PairwiseRankingLoss(nn.Module):
    """
    Response Ranking Learning (§3.2 of RERUM).
    Penalizes pairs where the relative ordering of predicted revenues
    is inconsistent with the actual revenues.
    """
    def _group_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if len(y) < 2:
            return torch.tensor(0.0, device=y.device)
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)
        yhat_diff = y_hat.unsqueeze(1) - y_hat.unsqueeze(0)
        # Only penalise pairs where the ordering is reversed
        wrong_order = (y_diff * yhat_diff) < 0
        penalty = (yhat_diff - y_diff).abs() * wrong_order.float()
        n_pairs = wrong_order.float().sum().clamp(min=1.0)
        return penalty.sum() / n_pairs

    def forward(self, y_hat_t, y_t, y_hat_c, y_c):
        """
        Parameters
        ----------
        y_hat_t, y_t : predicted and true outcomes for treated
        y_hat_c, y_c : predicted and true outcomes for control
        """
        loss_t = self._group_loss(y_hat_t, y_t)
        loss_c = self._group_loss(y_hat_c, y_c)
        return (loss_t + loss_c) / 2.0


class ListwiseUpliftLoss(nn.Module):
    """
    Uplift Ranking Learning (§3.3 of RERUM).
    Optimizes the global ranking of responders using a softmax-based
    cross-entropy loss over the predicted uplifts.
    """
    def forward(self, tau_hat, y1, y0, treatment):
        """
        Parameters
        ----------
        tau_hat   : (B,) predicted uplift
        y1, y0    : potential outcomes (labels used as proxies)
        treatment : (B,) treatment indicator
        """
        log_softmax_tau = F.log_softmax(tau_hat, dim=0)
        mask_t = (treatment == 1)
        mask_c = (treatment == 0)
        loss = torch.tensor(0.0, device=tau_hat.device)
        if mask_t.sum() > 0:
            treated_term = (y1[mask_t] * log_softmax_tau[mask_t]).mean()
            loss = loss - treated_term
        if mask_c.sum() > 0:
            control_term = (y0[mask_c] * log_softmax_tau[mask_c]).mean()
            loss = loss + control_term
        return loss

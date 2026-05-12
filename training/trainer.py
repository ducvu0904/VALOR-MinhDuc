"""
Trainers for baseline and VALOR models.

 • BaselineTrainer  — trains any baseline (MSE or ZILN loss)
 • VALORTrainer     — trains VALOR models with full Focal-ZILN + ranking
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.losses import (
    ZILNLoss, FocalZILNLoss, VALORLoss,
    PairwiseRankingLoss, ListwiseUpliftLoss
)
from utils.ziln_utils import ziln_expected_value, clamp_sigma

LOG_2PI = math.log(2 * math.pi)


# =====================================================================
#  Baseline Trainer
# =====================================================================

class BaselineTrainer:
    """
    Trains standard baseline models (TARNet, DragonNet, CFR, UniTE, EUEN, TLearner).

    Supports both MSE (``use_ziln=False``) and ZILN loss (``use_ziln=True``)
    modes. When ``use_focal=True`` and ``use_ziln=True``, uses Focal-ZILN
    instead of standard ZILN.

    Parameters
    ----------
    model       : nn.Module
    lr          : float
    epochs      : int
    device      : str
    use_focal   : bool  — use Focal-ZILN instead of standard ZILN
    focal_gamma : float
    focal_alpha : float
    lambda_ipm  : float — weight for IPM loss (CFR only)
    lambda_prop : float — weight for propensity loss (DragonNet only)
    """

    def __init__(
        self,
        model,
        lr: float = 5e-4,
        epochs: int = 30,
        device: str = "cpu",
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        lambda_ipm: float = 1.0,
        lambda_prop: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lambda_ipm = lambda_ipm
        self.lambda_prop = lambda_prop

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Determine loss function
        use_ziln = getattr(model, "use_ziln", False)
        if use_ziln:
            if use_focal:
                self.loss_fn = FocalZILNLoss(gamma=focal_gamma, alpha=focal_alpha)
            else:
                self.loss_fn = ZILNLoss()
        else:
            self.loss_fn = nn.MSELoss()

        self.use_ziln = use_ziln

    def _compute_loss(self, model_out, treatment, label):
        """Compute per-batch loss based on model type."""
        model_name = type(self.model).__name__
        total_loss = torch.tensor(0.0, device=self.device)

        if model_name == "SLearner":
            # SLearner forward already returns scalar
            pred = model_out
            total_loss = self.loss_fn(pred, label)
            return total_loss

        # Unpack outputs based on model type
        if model_name == "DragonNet":
            y0, y1, prop_logit, eps = model_out
        elif model_name == "CFR":
            y0, y1, ipm_loss = model_out
        elif model_name == "UniTE":
            mu_prog, tau = model_out[:2]
            y0, y1 = mu_prog, tau   # kept for ZILN path below
        elif model_name == "EUEN":
            control, uplift = model_out[:2]
            y0, y1 = control, uplift  # kept for ZILN path below
        else:
            y0, y1 = model_out[:2]

        t = treatment
        mask_t = (t == 1)
        mask_c = (t == 0)

        if self.use_ziln:
            # y0, y1 are each (pi_logit, mu, sigma)
            pi0, mu0, sig0 = y0
            pi1, mu1, sig1 = y1

            # Factual loss: compute ZILN loss on the observed arm
            if mask_c.sum() > 0:
                loss_c = self.loss_fn(
                    pi0[mask_c], mu0[mask_c], sig0[mask_c], label[mask_c]
                )
            else:
                loss_c = torch.tensor(0.0, device=self.device)

            if mask_t.sum() > 0:
                loss_t = self.loss_fn(
                    pi1[mask_t], mu1[mask_t], sig1[mask_t], label[mask_t]
                )
            else:
                loss_t = torch.tensor(0.0, device=self.device)

            total_loss = loss_c + loss_t
        else:
            # MSE on factual outcomes
            if model_name == "UniTE":
                # mu_prog + T*tau is the factual prediction
                mu_prog, tau = y0, y1
                pred = mu_prog + t.float() * tau
            elif model_name == "EUEN":
                # control + T*uplift is the factual prediction
                control, uplift = y0, y1
                pred = control + t.float() * uplift
            else:
                pred = torch.where(mask_t, y1, y0)
            total_loss = self.loss_fn(pred, label)

        # DragonNet propensity loss and targeted regularization
        if model_name == "DragonNet":
            prop_loss = F.binary_cross_entropy_with_logits(
                prop_logit, treatment, reduction="mean"
            )
            
            # Targeted regularization
            t_pred = torch.sigmoid(prop_logit)
            t_pred_clipped = (t_pred + 0.01) / 1.02
            
            if self.use_ziln:
                y0_pred = ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
                y1_pred = ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
            else:
                y0_pred, y1_pred = y0, y1
                
            t_float = t.float()
            y_pred = t_float * y1_pred + (1 - t_float) * y0_pred
            h = (t_float / t_pred_clipped) - ((1 - t_float) / (1 - t_pred_clipped))
            y_pert = y_pred + eps * h

            label_std = label.std().clamp(min=1.0)
            targeted_regularization = torch.mean((label - y_pert)**2)
            
            total_loss = total_loss + prop_loss

        # CFR IPM loss
        if model_name == "CFR" and ipm_loss is not None:
            total_loss = total_loss + self.lambda_ipm * ipm_loss

        return total_loss

    def train(self, train_loader, val_loader=None):
        """Full training loop. Restores best-val-loss checkpoint at the end."""
        model = self.model
        history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x_cat, x_num, treatment, label, true_tau = [
                    b.to(self.device) for b in batch
                ]

                self.optimizer.zero_grad()

                model_name = type(model).__name__

                if model_name == "SLearner":
                    out = model(x_cat, x_num, treatment)
                elif model_name == "CFR":
                    out = model(x_cat, x_num, treatment)
                else:
                    out = model(x_cat, x_num)

                loss = self._compute_loss(out, treatment, label)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # Validation + checkpointing
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader)
                history["val_loss"].append(val_loss)
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
            else:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        # Restore best checkpoint
        model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def _eval_loss(self, loader):
        self.model.eval()
        total = 0.0
        n = 0
        model_name = type(self.model).__name__

        for batch in loader:
            x_cat, x_num, treatment, label, true_tau = [
                b.to(self.device) for b in batch
            ]
            if model_name == "SLearner":
                out = self.model(x_cat, x_num, treatment)
            elif model_name == "CFR":
                out = self.model(x_cat, x_num, treatment)
            else:
                out = self.model(x_cat, x_num)

            loss = self._compute_loss(out, treatment, label)
            total += loss.item()
            n += 1
        return total / max(n, 1)


# =====================================================================
#  VALOR Trainer
# =====================================================================

class VALORTrainer:
    """
    Trains VALOR models with Focal-ZILN + optional ValueWeightedRanking.

    Parameters
    ----------
    model        : VALOR instance
    lr           : float
    epochs       : int
    device       : str
    gamma, alpha : focal hyper-params
    lambda_rank  : weight for ranking loss
    use_ranking  : bool — whether to include ranking loss
    """

    def __init__(
        self,
        model,
        lr: float = 5e-4,
        epochs: int = 30,
        device: str = "cpu",
        gamma: float = 2.0,
        alpha: float = 0.25,
        lambda_rank: float = 1.0,
        use_ranking: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.use_ranking = use_ranking

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if use_ranking:
            self.loss_fn = VALORLoss(
                gamma=gamma, alpha=alpha, lambda_rank=lambda_rank
            )
            # Store FocalZILN ref for re-use in the training loop (Bug 3 fix)
            self._focal_ziln = self.loss_fn.focal_ziln
        else:
            self.loss_fn = FocalZILNLoss(gamma=gamma, alpha=alpha)
            self._focal_ziln = self.loss_fn

        # A separate instance for validation (no ranking, no grad)
        self._val_focal_ziln = FocalZILNLoss(gamma=gamma, alpha=alpha)

    def train(self, train_loader, val_loader=None):
        """Full training loop. Restores best-val-loss checkpoint at the end."""
        model = self.model
        history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x_cat, x_num, treatment, label, true_tau = [
                    b.to(self.device) for b in batch
                ]

                self.optimizer.zero_grad()

                y0_params, y1_params, extras = model(x_cat, x_num, treatment)
                pi0, mu0, sig0 = y0_params
                pi1, mu1, sig1 = y1_params

                # Select factual arm
                t = treatment
                mask_t = (t == 1)
                mask_c = (t == 0)

                # Factual ZILN losses
                loss_parts = torch.tensor(0.0, device=self.device)

                if self.use_ranking:
                    # Compute factual loss using the stored _focal_ziln (Bug 3 fix)
                    if mask_c.sum() > 0:
                        loss_c = self._focal_ziln(
                            pi0[mask_c], mu0[mask_c], sig0[mask_c], label[mask_c]
                        )
                    else:
                        loss_c = torch.tensor(0.0, device=self.device)

                    if mask_t.sum() > 0:
                        loss_t = self._focal_ziln(
                            pi1[mask_t], mu1[mask_t], sig1[mask_t], label[mask_t]
                        )
                    else:
                        loss_t = torch.tensor(0.0, device=self.device)

                    l_fl_ziln = loss_c + loss_t

                    # Predicted uplift for ranking — needs gradients (Bug 2 fix: removed dead no_grad block)
                    tau_hat = (
                        ziln_expected_value(torch.sigmoid(pi1), mu1, sig1)
                        - ziln_expected_value(torch.sigmoid(pi0), mu0, sig0)
                    )
                    z = (2.0 * treatment.float() - 1.0) * label  # sign-adjusted transformed outcome

                    ranking_loss = self.loss_fn.ranking(tau_hat, z)
                    total_loss = (
                        l_fl_ziln
                        + self.loss_fn.lambda_rank * ranking_loss
                    )

                    # IPM if present
                    if "ipm_loss" in extras and extras["ipm_loss"] is not None:
                        total_loss = total_loss + extras.get("lambda_ipm", 1.0) * extras["ipm_loss"]

                    loss = total_loss
                else:
                    # Focal-ZILN only (no ranking)
                    if mask_c.sum() > 0:
                        loss_c = self.loss_fn(
                            pi0[mask_c], mu0[mask_c], sig0[mask_c], label[mask_c]
                        )
                    else:
                        loss_c = torch.tensor(0.0, device=self.device)

                    if mask_t.sum() > 0:
                        loss_t = self.loss_fn(
                            pi1[mask_t], mu1[mask_t], sig1[mask_t], label[mask_t]
                        )
                    else:
                        loss_t = torch.tensor(0.0, device=self.device)

                    loss = loss_c + loss_t

                    if "ipm_loss" in extras and extras["ipm_loss"] is not None:
                        loss = loss + extras.get("lambda_ipm", 1.0) * extras["ipm_loss"]

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # Validation + checkpointing
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader)
                history["val_loss"].append(val_loss)
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
            else:
                print(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        # Restore best checkpoint
        model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def _eval_loss(self, loader):
        """Validation loss (Focal-ZILN only, no ranking for speed)."""
        self.model.eval()
        total = 0.0
        n = 0

        for batch in loader:
            x_cat, x_num, treatment, label, true_tau = [
                b.to(self.device) for b in batch
            ]
            y0_params, y1_params, extras = self.model(x_cat, x_num, treatment)
            pi0, mu0, sig0 = y0_params
            pi1, mu1, sig1 = y1_params

            mask_t = (treatment == 1)
            mask_c = (treatment == 0)

            loss = torch.tensor(0.0, device=self.device)
            if mask_c.sum() > 0:
                loss = loss + self._val_focal_ziln(
                    pi0[mask_c], mu0[mask_c], sig0[mask_c], label[mask_c]
                )
            if mask_t.sum() > 0:
                loss = loss + self._val_focal_ziln(
                    pi1[mask_t], mu1[mask_t], sig1[mask_t], label[mask_t]
                )

            total += loss.item()
            n += 1

        return total / max(n, 1)


# =====================================================================
#  RERUM Trainer
# =====================================================================

class RERUMTrainer:
    """
    Trains a RERUMWrapper model with ZILN + pairwise + listwise losses.
    """
    def __init__(
        self, model, lr=5e-4, epochs=30, device="cpu",
        lambda_rank=1.0, lambda_lu=1.0, lambda_ipm=1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lambda_rank = lambda_rank
        self.lambda_lu = lambda_lu
        self.lambda_ipm = lambda_ipm

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._ziln = ZILNLoss()
        self._pair = PairwiseRankingLoss()
        self._listwise = ListwiseUpliftLoss()

    def train(self, train_loader, val_loader=None):
        model = self.model
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(self.epochs):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            for batch in train_loader:
                x_cat, x_num, treatment, label, _ = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                y0_p, y1_p, extras = model.forward_with_treatment(x_cat, x_num, treatment)
                
                # ZILN
                mask_t, mask_c = (treatment == 1), (treatment == 0)
                loss_ziln = torch.tensor(0.0, device=self.device)
                if mask_c.sum() > 0:
                    loss_ziln += self._ziln(y0_p[0][mask_c], y0_p[1][mask_c], y0_p[2][mask_c], label[mask_c])
                if mask_t.sum() > 0:
                    loss_ziln += self._ziln(y1_p[0][mask_t], y1_p[1][mask_t], y1_p[2][mask_t], label[mask_t])
                
                # Ranking
                y_h0 = ziln_expected_value(torch.sigmoid(y0_p[0]), y0_p[1], y0_p[2])
                y_h1 = ziln_expected_value(torch.sigmoid(y1_p[0]), y1_p[1], y1_p[2])
                tau_h = y_h1 - y_h0
                loss_p = self._pair(y_h1[mask_t], label[mask_t], y_h0[mask_c], label[mask_c]) if (mask_t.sum()>1 or mask_c.sum()>1) else torch.tensor(0.0, device=self.device)
                loss_l = self._listwise(tau_h, label, label, treatment)
                
                total_loss = loss_ziln + self.lambda_rank * loss_p + self.lambda_lu * loss_l
                if "ipm_loss" in extras and extras["ipm_loss"] is not None:
                    total_loss += extras.get("lambda_ipm", self.lambda_ipm) * extras["ipm_loss"]
                
                total_loss.backward()
                self.optimizer.step()
                epoch_loss += total_loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)
            if val_loader:
                val_loss = self._eval_loss(val_loader)
                history["val_loss"].append(val_loss)
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            else:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train:.4f}")

        model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def _eval_loss(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            x_cat, x_num, treatment, label, _ = [b.to(self.device) for b in batch]
            y0_p, y1_p, extras = self.model.forward_with_treatment(x_cat, x_num, treatment)
            mask_t, mask_c = (treatment == 1), (treatment == 0)
            
            loss_ziln = torch.tensor(0.0, device=self.device)
            if mask_c.sum() > 0:
                loss_ziln += self._ziln(y0_p[0][mask_c], y0_p[1][mask_c], y0_p[2][mask_c], label[mask_c])
            if mask_t.sum() > 0:
                loss_ziln += self._ziln(y1_p[0][mask_t], y1_p[1][mask_t], y1_p[2][mask_t], label[mask_t])
            
            y_h0 = ziln_expected_value(torch.sigmoid(y0_p[0]), y0_p[1], y0_p[2])
            y_h1 = ziln_expected_value(torch.sigmoid(y1_p[0]), y1_p[1], y1_p[2])
            tau_h = y_h1 - y_h0
            loss_p = self._pair(y_h1[mask_t], label[mask_t], y_h0[mask_c], label[mask_c]) if (mask_t.sum()>1 or mask_c.sum()>1) else torch.tensor(0.0, device=self.device)
            loss_l = self._listwise(tau_h, label, label, treatment)
            
            total_loss = loss_ziln + self.lambda_rank * loss_p + self.lambda_lu * loss_l
            if "ipm_loss" in extras and extras["ipm_loss"] is not None:
                total_loss += extras.get("lambda_ipm", self.lambda_ipm) * extras["ipm_loss"]
                
            total += total_loss.item()
            n += 1
        return total / max(n, 1)

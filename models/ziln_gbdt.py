"""
Robust ZILN-GBDT  —  tree-based uplift variant (Appendix B).

Standalone implementation — does NOT share code with the DNN path.

Key ideas:
 - Custom split criterion: maximise Euclidean distance of ZILN uplift
   between child nodes.
 - Adaptive Bayesian Smoothing to prevent variance collapse in sparse
   leaf nodes (Algorithm 1 in Appendix B).
 - Uses ``ziln_utils`` for the E[y] formula shared with the DNN heads.
"""

import numpy as np
from utils.ziln_utils import (
    ziln_expected_value_np,
    ziln_uplift_np,
    clamp_sigma_np,
    SIGMA_MIN,
    SIGMA_MAX,
)


# =====================================================================
#  ZILN parameter estimation with Bayesian smoothing
# =====================================================================

def _estimate_ziln_params(y, alpha_reg: float = 10.0,
                          global_p=None, global_mu=None, global_sigma=None):
    """
    Estimate smoothed ZILN parameters (p, μ, σ) from observed outcomes.

    Applies Adaptive Bayesian Smoothing (Algorithm 1):
        p_s  = w·p_leaf  + (1−w)·p̄
        μ_s  = w·μ_leaf  + (1−w)·μ̄
        σ_s  = w·σ_leaf  + (1−w)·σ̄
    where w = n_pos / (n_pos + α_reg).
    """
    n = len(y)
    if n == 0:
        return 0.0, 0.0, SIGMA_MIN

    pos_mask = y > 0
    n_pos = pos_mask.sum()

    # Raw leaf estimates
    p_leaf = n_pos / n if n > 0 else 0.0
    if n_pos > 0:
        log_y = np.log(y[pos_mask])
        mu_leaf = log_y.mean()
        sigma_leaf = log_y.std() if n_pos > 1 else SIGMA_MIN
    else:
        mu_leaf = 0.0
        sigma_leaf = SIGMA_MIN

    # Smoothing weight
    w = n_pos / (n_pos + alpha_reg)

    # Apply smoothing (fall back to raw if priors not provided)
    if global_p is not None:
        p_s = w * p_leaf + (1 - w) * global_p
    else:
        p_s = p_leaf

    if global_mu is not None:
        mu_s = w * mu_leaf + (1 - w) * global_mu
    else:
        mu_s = mu_leaf

    if global_sigma is not None:
        sigma_s = w * sigma_leaf + (1 - w) * global_sigma
    else:
        sigma_s = sigma_leaf

    sigma_s = float(np.clip(sigma_s, SIGMA_MIN, SIGMA_MAX))
    return float(p_s), float(mu_s), float(sigma_s)


def _ziln_ev(p, mu, sigma):
    """Scalar E[Y] = p · exp(μ + σ²/2)."""
    return p * np.exp(mu + sigma ** 2 / 2.0)


# =====================================================================
#  Single uplift tree
# =====================================================================

class ZILNUpliftTree:
    """
    A single uplift decision tree with ZILN-based splitting.

    Split criterion — maximise ZILN Uplift Heterogeneity Gain:
        Gain(s) = (N_L · N_R / (N_L + N_R)²) · (τ̂_L − τ̂_R)²
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 50,
        alpha_reg: float = 10.0,
        max_features: float = 1.0,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha_reg = alpha_reg
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self.tree = None

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------
    def fit(self, X, treatment, y):
        """
        Parameters
        ----------
        X         : (N, d) ndarray
        treatment : (N,) ndarray  0/1
        y         : (N,) ndarray  ≥ 0
        """
        # Compute global priors from the full training set
        self._compute_global_priors(y, treatment)
        indices = np.arange(len(y))
        self.tree = self._build_node(X, treatment, y, indices, depth=0)
        return self

    def _compute_global_priors(self, y, treatment):
        """Compute global ZILN priors for both arms."""
        for arm in (0, 1):
            mask = treatment == arm
            ya = y[mask]
            pos = ya[ya > 0]
            p = (ya > 0).mean() if len(ya) > 0 else 0.0
            mu = np.log(pos).mean() if len(pos) > 0 else 0.0
            sigma = np.log(pos).std() if len(pos) > 1 else SIGMA_MIN
            sigma = float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
            setattr(self, f"_global_p_{arm}", p)
            setattr(self, f"_global_mu_{arm}", mu)
            setattr(self, f"_global_sigma_{arm}", sigma)

    def _estimate_uplift(self, y, treatment, indices):
        """Estimate ZILN uplift in a node."""
        idx = indices
        t = treatment[idx]
        ya = y[idx]

        params = {}
        for arm in (0, 1):
            arm_mask = t == arm
            params[arm] = _estimate_ziln_params(
                ya[arm_mask],
                alpha_reg=self.alpha_reg,
                global_p=getattr(self, f"_global_p_{arm}"),
                global_mu=getattr(self, f"_global_mu_{arm}"),
                global_sigma=getattr(self, f"_global_sigma_{arm}"),
            )

        p1, mu1, sig1 = params[1]
        p0, mu0, sig0 = params[0]
        tau = _ziln_ev(p1, mu1, sig1) - _ziln_ev(p0, mu0, sig0)
        return tau

    # ------------------------------------------------------------------
    def _build_node(self, X, treatment, y, indices, depth):
        """Recursively build the tree."""
        tau = self._estimate_uplift(y, treatment, indices)

        # Stopping conditions
        if (
            depth >= self.max_depth
            or len(indices) < 2 * self.min_samples_leaf
        ):
            return {"leaf": True, "tau": tau, "n": len(indices)}

        best_gain = -np.inf
        best_split = None

        n_features = X.shape[1]
        n_try = max(1, int(n_features * self.max_features))
        feature_subset = self.rng.choice(n_features, n_try, replace=False)

        for feat_idx in feature_subset:
            vals = X[indices, feat_idx]
            unique_vals = np.unique(vals)
            if len(unique_vals) <= 1:
                continue

            # Try quantile-based thresholds for efficiency
            thresholds = np.percentile(vals, np.arange(10, 100, 10))
            thresholds = np.unique(thresholds)

            for thr in thresholds:
                left_mask = vals <= thr
                right_mask = ~left_mask
                n_l = left_mask.sum()
                n_r = right_mask.sum()

                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                left_idx = indices[left_mask]
                right_idx = indices[right_mask]

                tau_l = self._estimate_uplift(y, treatment, left_idx)
                tau_r = self._estimate_uplift(y, treatment, right_idx)

                # Gain = (N_L · N_R / (N_L + N_R)²) · (τ̂_L − τ̂_R)²
                gain = (n_l * n_r / (n_l + n_r) ** 2) * (tau_l - tau_r) ** 2

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature": feat_idx,
                        "threshold": thr,
                        "left_idx": left_idx,
                        "right_idx": right_idx,
                    }

        if best_split is None:
            return {"leaf": True, "tau": tau, "n": len(indices)}

        left_node = self._build_node(
            X, treatment, y, best_split["left_idx"], depth + 1
        )
        right_node = self._build_node(
            X, treatment, y, best_split["right_idx"], depth + 1
        )

        return {
            "leaf": False,
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_node,
            "right": right_node,
            "n": len(indices),
        }

    # ------------------------------------------------------------------
    #  Predict
    # ------------------------------------------------------------------
    def predict_uplift(self, X):
        """Return ZILN uplift estimate for each row."""
        return np.array([self._predict_single(X[i]) for i in range(len(X))])

    def _predict_single(self, x):
        node = self.tree
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["tau"]


# =====================================================================
#  Ensemble (bagged forest)
# =====================================================================

class ZILNGBDTForest:
    """
    Ensemble of ``ZILNUpliftTree`` with bagging.

    Parameters
    ----------
    n_estimators    : int    (default 20, per paper §5.2)
    max_depth       : int    (default 6)
    subsample_ratio : float  fraction of data per tree
    """

    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int = 6,
        min_samples_leaf: int = 50,
        alpha_reg: float = 10.0,
        subsample_ratio: float = 0.8,
        max_features: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha_reg = alpha_reg
        self.subsample_ratio = subsample_ratio
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, treatment, y):
        rng = np.random.RandomState(self.random_state)
        N = len(X)
        self.trees = []

        for i in range(self.n_estimators):
            # Bagging: sample with replacement
            idx = rng.choice(N, size=int(N * self.subsample_ratio), replace=True)
            tree = ZILNUpliftTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                alpha_reg=self.alpha_reg,
                max_features=self.max_features,
                random_state=self.random_state + i,
            )
            tree.fit(X[idx], treatment[idx], y[idx])
            self.trees.append(tree)

        return self

    def predict_uplift(self, X):
        """Average uplift across all trees."""
        preds = np.stack([t.predict_uplift(X) for t in self.trees], axis=0)
        return preds.mean(axis=0)

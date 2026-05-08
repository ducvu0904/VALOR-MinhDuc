"""
Treatment-Gated Interaction Module  (§4.3 of the paper).

Pure ``nn.Module`` — no loss or backbone dependency.

    Interaction(X, T) = (W_x · X + b_x) ⊙ σ(W_t · T + b_t)

where ⊙ is element-wise Hadamard product and σ is sigmoid.
This creates a differentiable, treatment-conditional feature selector
that explicitly "zeros-out" irrelevant feature subspaces.
"""

import torch
import torch.nn as nn


class TreatmentGatedInteraction(nn.Module):
    """
    Bilinear gating mechanism that dynamically re-weights a feature
    representation based on the treatment assignment.

    Parameters
    ----------
    input_dim  : int
        Dimensionality of the feature representation X.
    hidden_dim : int
        Output dimensionality of the gated representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Feature transform
        self.fc_x = nn.Linear(input_dim, hidden_dim)
        # Treatment embedding → gate (treatment is scalar 0/1)
        self.fc_t = nn.Linear(1, hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim)  feature representation
        t : (B,) or (B, 1)  treatment indicator (0 / 1)

        Returns
        -------
        (B, hidden_dim) gated representation
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (B, 1)
        t = t.float()

        h_x = self.fc_x(x)                # (B, hidden_dim)
        gate = torch.sigmoid(self.fc_t(t)) # (B, hidden_dim)
        return h_x * gate                  # Hadamard product

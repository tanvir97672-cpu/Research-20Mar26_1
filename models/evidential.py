"""
Evidential Deep Learning Head for DAOS-RFF.

Implements Dirichlet-based evidential classification for uncertainty quantification
in open-set RF fingerprinting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class EvidentialHead(nn.Module):
    """
    Evidential classification head using Dirichlet prior.

    Outputs evidence (Dirichlet concentration - 1), from which we compute:
    - Class probabilities (expected value of Dirichlet)
    - Uncertainty (vacuity based on total evidence)
    - Alpha parameters for Dirichlet distribution

    Reference: "Evidential Deep Learning to Quantify Classification Uncertainty"
    """

    def __init__(
        self,
        in_dim: int = 128,
        num_classes: int = 5,
        hidden_dim: int = 64,
    ):
        """
        Initialize evidential head.

        Args:
            in_dim: Input embedding dimension
            num_classes: Number of known classes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.num_classes = num_classes

        # MLP to predict evidence
        self.evidence_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute evidential outputs.

        Args:
            x: Input embedding tensor of shape (B, in_dim)

        Returns:
            Dictionary containing:
                - 'evidence': Evidence values (B, num_classes), non-negative
                - 'alpha': Dirichlet concentration parameters (B, num_classes)
                - 'S': Total evidence / Dirichlet strength (B, 1)
                - 'prob': Class probabilities (B, num_classes)
                - 'uncertainty': Vacuity-based uncertainty (B, 1)
                - 'logits': Raw logits before softplus (B, num_classes)
        """
        # Get raw logits
        logits = self.evidence_net(x)

        # Evidence must be non-negative: use softplus
        evidence = F.softplus(logits)

        # Dirichlet concentration parameters
        alpha = evidence + 1.0

        # Total evidence (Dirichlet strength)
        S = alpha.sum(dim=1, keepdim=True)

        # Class probabilities (expected value of Dirichlet)
        prob = alpha / S

        # Uncertainty: vacuity = K / S (high when total evidence is low)
        uncertainty = self.num_classes / S

        return {
            'evidence': evidence,
            'alpha': alpha,
            'S': S,
            'prob': prob,
            'uncertainty': uncertainty,
            'logits': logits,
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        out = self.forward(x)
        return out['prob'].argmax(dim=1)


def evidential_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    epoch: int = 0,
    num_classes: int = 5,
    annealing_step: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute evidential deep learning loss with numerical stability.

    Combines:
    1. Negative log-likelihood of Dirichlet for correct class
    2. KL divergence regularizer to avoid overconfident wrong predictions

    NOTE: Uses float32 internally for numerical stability with mixed precision.

    Args:
        alpha: Dirichlet concentration parameters (B, K)
        target: Ground truth class indices (B,)
        epoch: Current training epoch for annealing
        num_classes: Number of classes
        annealing_step: Epoch when KL weight reaches 1
        device: Computation device

    Returns:
        Scalar loss value
    """
    if device is None:
        device = alpha.device

    # Cast to float32 for numerical stability
    alpha = alpha.float()

    batch_size = alpha.shape[0]

    # One-hot encode target
    target_onehot = F.one_hot(target, num_classes=num_classes).float().to(device)

    # Dirichlet strength
    S = alpha.sum(dim=1, keepdim=True)

    # Clamp alpha and S to prevent log(0) and numerical instability
    alpha_clamped = torch.clamp(alpha, min=1e-6)
    S_clamped = torch.clamp(S, min=1e-6)

    # Loss 1: Negative log-likelihood (Type II MLE)
    # L_nll = sum_k y_k * (log(S) - log(alpha_k))
    loss_nll = torch.sum(
        target_onehot * (torch.log(S_clamped) - torch.log(alpha_clamped)),
        dim=1
    ).mean()

    # Loss 2: KL divergence regularizer
    # Encourages uniform distribution for incorrect predictions

    # Remove evidence for correct class
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha

    # Clamp for stability
    alpha_tilde = torch.clamp(alpha_tilde, min=1e-6)

    # KL divergence between alpha_tilde and uniform prior
    alpha_tilde_sum = alpha_tilde.sum(dim=1, keepdim=True)
    prior = torch.ones_like(alpha)
    prior_sum = prior.sum(dim=1, keepdim=True)

    kl_div = (
        torch.lgamma(alpha_tilde_sum) - torch.lgamma(prior_sum)
        - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        + torch.sum(torch.lgamma(prior), dim=1, keepdim=True)
        + torch.sum(
            (alpha_tilde - prior) * (
                torch.digamma(alpha_tilde) - torch.digamma(alpha_tilde_sum)
            ),
            dim=1, keepdim=True
        )
    ).mean()

    # Clamp KL divergence to prevent extreme values
    kl_div = torch.clamp(kl_div, min=0.0, max=100.0)

    # Annealing coefficient for KL term
    annealing_coef = min(1.0, epoch / annealing_step)

    # Total loss
    loss = loss_nll + annealing_coef * kl_div

    return loss

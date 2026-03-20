"""
Adversarial Components for DAOS-RFF.

Includes:
- Gradient Reversal Layer (GRL)
- Channel Adversary (for channel-invariant features)
- Domain Adversary (for domain adaptation)
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer function.

    Forward pass: identity
    Backward pass: negate gradients and scale by lambda
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.

    Used for domain-adversarial training to learn invariant features.
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Initialize GRL.

        Args:
            lambda_: Gradient scaling factor (can be scheduled)
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update lambda value (for scheduling)."""
        self.lambda_ = lambda_


class ChannelAdversary(nn.Module):
    """
    Channel condition classifier with gradient reversal.

    Predicts channel condition (e.g., indoor/outdoor/urban).
    Gradient reversal encourages the backbone to learn channel-invariant features.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        num_channels: int = 3,
        lambda_: float = 1.0,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize channel adversary.

        Args:
            in_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_channels: Number of channel conditions to classify
            lambda_: Gradient reversal scaling factor
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.grl = GradientReversalLayer(lambda_=lambda_)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embedding (B, in_dim)

        Returns:
            Channel logits (B, num_channels)
        """
        x_reversed = self.grl(x)
        return self.classifier(x_reversed)

    def set_lambda(self, lambda_: float):
        """Update GRL lambda."""
        self.grl.set_lambda(lambda_)


class DomainAdversary(nn.Module):
    """
    Domain classifier with gradient reversal.

    Predicts source vs target domain.
    Gradient reversal encourages domain-invariant features.
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        num_domains: int = 2,
        lambda_: float = 1.0,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize domain adversary.

        Args:
            in_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_domains: Number of domains (typically 2: source/target)
            lambda_: Gradient reversal scaling factor
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.grl = GradientReversalLayer(lambda_=lambda_)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embedding (B, in_dim)

        Returns:
            Domain logits (B, num_domains)
        """
        x_reversed = self.grl(x)
        return self.classifier(x_reversed)

    def set_lambda(self, lambda_: float):
        """Update GRL lambda."""
        self.grl.set_lambda(lambda_)


def compute_grl_lambda(
    epoch: int,
    max_epochs: int,
    gamma: float = 10.0,
) -> float:
    """
    Compute GRL lambda using schedule from DANN paper.

    Lambda increases from 0 to 1 following a sigmoid schedule.

    Args:
        epoch: Current epoch
        max_epochs: Total epochs
        gamma: Steepness of schedule

    Returns:
        Lambda value for current epoch
    """
    p = epoch / max_epochs
    lambda_ = 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0
    return lambda_

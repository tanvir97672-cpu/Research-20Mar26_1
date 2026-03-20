"""
Loss functions for DAOS-RFF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def evidential_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    epoch: int = 0,
    num_classes: int = 5,
    annealing_step: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute evidential deep learning loss.

    Combines:
    1. Negative log-likelihood of Dirichlet for correct class
    2. KL divergence regularizer

    Args:
        alpha: Dirichlet concentration parameters (B, K)
        target: Ground truth class indices (B,)
        epoch: Current epoch for annealing
        num_classes: Number of classes
        annealing_step: Epoch when KL weight reaches 1
        device: Computation device

    Returns:
        Loss value
    """
    if device is None:
        device = alpha.device

    # One-hot encode target
    target_onehot = F.one_hot(target, num_classes=num_classes).float().to(device)

    # Dirichlet strength
    S = alpha.sum(dim=1, keepdim=True)

    # NLL loss
    loss_nll = torch.sum(
        target_onehot * (torch.log(S) - torch.log(alpha)),
        dim=1
    ).mean()

    # KL divergence regularizer
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha
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

    annealing_coef = min(1.0, epoch / annealing_step)
    loss = loss_nll + annealing_coef * kl_div

    return loss


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Supervised contrastive loss.

    Args:
        embeddings: Feature embeddings (B, D)
        labels: Class labels (B,)
        temperature: Temperature scaling

    Returns:
        Loss value
    """
    device = embeddings.device

    if embeddings.shape[0] < 2:
        return torch.tensor(0.0, device=device)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

    # Positive mask (same class, excluding self)
    labels = labels.unsqueeze(0)
    pos_mask = (labels == labels.t()).float()
    pos_mask.fill_diagonal_(0)

    # Numerical stability
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    exp_sim = torch.exp(sim_matrix)

    # All pairs mask (excluding self)
    all_mask = torch.ones_like(sim_matrix)
    all_mask.fill_diagonal_(0)

    # Loss computation
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    all_sim = (exp_sim * all_mask).sum(dim=1)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0

    if not valid.any():
        return torch.tensor(0.0, device=device)

    loss = -torch.log(pos_sim[valid] / (all_sim[valid] + 1e-8) + 1e-8)
    loss = loss.mean()

    return loss

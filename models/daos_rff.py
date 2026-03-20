"""
DAOS-RFF: Domain-Adaptive Open-Set RF Fingerprinting.

Main PyTorch Lightning module implementing the complete DAOS-RFF framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from typing import Dict, Optional, Tuple, Any

from .backbone import SharedBackbone
from .evidential import EvidentialHead, evidential_loss
from .adversarial import (
    ChannelAdversary,
    DomainAdversary,
    compute_grl_lambda,
)


class DAOS_RFF(pl.LightningModule):
    """
    Domain-Adaptive Open-Set RF Fingerprinting (DAOS-RFF).

    Combines:
    1. Shared backbone for feature extraction
    2. Evidential head for uncertainty quantification
    3. Channel adversary for channel-invariant features
    4. Domain adversary for domain adaptation
    5. Supervised contrastive learning (optional)
    """

    def __init__(
        self,
        # Backbone config
        pretrained: bool = False,
        feature_dim: int = 512,
        embedding_dim: int = 128,
        # Evidential config
        num_classes: int = 5,
        # Adversarial config
        num_channels: int = 3,
        num_domains: int = 2,
        # Loss weights
        lambda_adv_channel: float = 0.1,
        lambda_adv_domain: float = 0.1,
        lambda_contrastive: float = 0.1,
        # Training config
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        warmup_epochs: int = 5,
        # Dropout
        dropout_rate: float = 0.3,
        # Other
        kl_annealing_step: int = 10,
    ):
        """
        Initialize DAOS-RFF model.

        Args:
            pretrained: Use pretrained backbone
            feature_dim: Backbone feature dimension
            embedding_dim: Output embedding dimension
            num_classes: Number of known device classes
            num_channels: Number of channel conditions
            num_domains: Number of domains
            lambda_adv_channel: Channel adversary loss weight
            lambda_adv_domain: Domain adversary loss weight
            lambda_contrastive: Contrastive loss weight
            learning_rate: Initial learning rate
            weight_decay: Weight decay
            max_epochs: Total training epochs
            warmup_epochs: Warmup epochs
            dropout_rate: Dropout rate
            kl_annealing_step: KL annealing step for evidential loss
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Build model components
        self.backbone = SharedBackbone(
            pretrained=pretrained,
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )

        self.evidential_head = EvidentialHead(
            in_dim=embedding_dim,
            num_classes=num_classes,
        )

        self.channel_adversary = ChannelAdversary(
            in_dim=embedding_dim,
            num_channels=num_channels,
            dropout_rate=dropout_rate,
        )

        self.domain_adversary = DomainAdversary(
            in_dim=embedding_dim,
            num_domains=num_domains,
            dropout_rate=dropout_rate,
        )

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="binary")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input spectrogram (B, 1, H, W)

        Returns:
            Dictionary with all outputs
        """
        # Extract features
        backbone_out = self.backbone(x)
        embedding = backbone_out['embedding']
        features = backbone_out['features']

        # Evidential classification
        evidential_out = self.evidential_head(embedding)

        # Adversarial predictions
        channel_logits = self.channel_adversary(embedding)
        domain_logits = self.domain_adversary(embedding)

        return {
            'embedding': embedding,
            'features': features,
            **evidential_out,
            'channel_logits': channel_logits,
            'domain_logits': domain_logits,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with numerical stability for mixed precision."""
        x = batch['spectrogram']
        device_label = batch['device_label']
        channel_label = batch['channel_label']
        domain_label = batch['domain_label']
        is_known = batch['is_known']

        # Only use known samples for classification training
        known_mask = is_known

        if not known_mask.any():
            # No known samples in batch, skip
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        # Forward pass on all samples
        outputs = self(x)

        # Get known sample outputs
        known_indices = known_mask.nonzero(as_tuple=True)[0]

        # 1. Evidential loss (only on known samples)
        alpha_known = outputs['alpha'][known_mask]
        labels_known = device_label[known_mask]

        loss_edl = evidential_loss(
            alpha=alpha_known,
            target=labels_known,
            epoch=self.current_epoch,
            num_classes=self.hparams.num_classes,
            annealing_step=self.hparams.kl_annealing_step,
            device=self.device,
        )

        # 2. Channel adversary loss (on all samples) - with stability
        loss_channel = F.cross_entropy(
            outputs['channel_logits'].float(),  # Cast to float32 for stability
            channel_label,
        )

        # 3. Domain adversary loss (on all samples) - with stability
        loss_domain = F.cross_entropy(
            outputs['domain_logits'].float(),  # Cast to float32 for stability
            domain_label,
        )

        # 4. Contrastive loss (on known samples)
        loss_con = self._contrastive_loss(
            outputs['embedding'][known_mask],
            labels_known,
        )

        # Update adversary lambda based on epoch
        grl_lambda = compute_grl_lambda(
            self.current_epoch,
            self.hparams.max_epochs,
        )
        self.channel_adversary.set_lambda(grl_lambda)
        self.domain_adversary.set_lambda(grl_lambda)

        # Combined loss with gradient scaling for stability
        loss = (
            loss_edl
            + self.hparams.lambda_adv_channel * loss_channel
            + self.hparams.lambda_adv_domain * loss_domain
            + self.hparams.lambda_contrastive * loss_con
        )

        # Clamp loss to prevent NaN/Inf
        loss = torch.clamp(loss, max=100.0)

        # Compute accuracy
        preds = outputs['prob'][known_mask].argmax(dim=1)
        self.train_acc(preds, labels_known)

        # Logging
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/loss_edl', loss_edl)
        self.log('train/loss_channel', loss_channel)
        self.log('train/loss_domain', loss_domain)
        self.log('train/loss_con', loss_con)
        self.log('train/acc', self.train_acc, prog_bar=True)
        self.log('train/grl_lambda', grl_lambda)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        x = batch['spectrogram']
        device_label = batch['device_label']
        is_known = batch['is_known']

        outputs = self(x)

        # Closed-set accuracy (on known samples)
        known_mask = is_known
        if known_mask.any():
            preds = outputs['prob'][known_mask].argmax(dim=1)
            labels_known = device_label[known_mask]
            self.val_acc(preds, labels_known)
            self.log('val/acc', self.val_acc, prog_bar=True)

        # Open-set detection (all samples)
        # Use uncertainty as score for detecting unknown
        uncertainty = outputs['uncertainty'].squeeze()

        # Binary labels: 1 = unknown, 0 = known
        is_unknown = ~is_known

        if is_unknown.any() and known_mask.any():
            self.val_auroc(uncertainty, is_unknown.long())
            self.log('val/auroc', self.val_auroc, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step with detailed metrics."""
        x = batch['spectrogram']
        device_label = batch['device_label']
        is_known = batch['is_known']

        outputs = self(x)

        # Store for epoch-end computation
        return {
            'uncertainty': outputs['uncertainty'].squeeze(),
            'prob': outputs['prob'],
            'device_label': device_label,
            'is_known': is_known,
        }

    def predict_with_rejection(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with open-set rejection.

        Args:
            x: Input spectrogram
            threshold: Uncertainty threshold for rejection

        Returns:
            predictions: Class predictions (-1 for rejected/unknown)
            uncertainty: Uncertainty scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)

            probs = outputs['prob']
            uncertainty = outputs['uncertainty'].squeeze()

            predictions = probs.argmax(dim=1)

            # Reject high-uncertainty samples
            is_rogue = uncertainty > threshold
            predictions[is_rogue] = -1

            return predictions, uncertainty

    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Simplified supervised contrastive loss.

        Args:
            embeddings: Feature embeddings (B, D)
            labels: Class labels (B,)
            temperature: Temperature scaling

        Returns:
            Contrastive loss value
        """
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=self.device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(0)
        pos_mask = (labels == labels.t()).float()

        # Remove diagonal
        pos_mask.fill_diagonal_(0)

        # Compute loss
        # For numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        exp_sim = torch.exp(sim_matrix)

        # Mask out self-similarity
        mask = torch.ones_like(sim_matrix)
        mask.fill_diagonal_(0)

        # Compute log-likelihood for positive pairs
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        all_sim = (exp_sim * mask).sum(dim=1)

        # Avoid division by zero
        pos_count = pos_mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1)

        loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
        loss = (loss * (pos_count > 0).float()).sum() / torch.clamp(
            (pos_count > 0).sum(), min=1
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

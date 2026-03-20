"""
Shared Backbone for DAOS-RFF.

ResNet18-based feature extractor for single-channel spectrograms.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional


class SharedBackbone(nn.Module):
    """
    Shared feature extraction backbone based on ResNet18.

    Modified for single-channel spectrogram input (1, H, W).
    Outputs feature embedding and intermediate features for adversarial branches.
    """

    def __init__(
        self,
        pretrained: bool = False,
        feature_dim: int = 512,
        embedding_dim: int = 128,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize backbone.

        Args:
            pretrained: Whether to use pretrained ImageNet weights
            feature_dim: Dimension of ResNet features (default 512 for ResNet18)
            embedding_dim: Output embedding dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        # Load ResNet18
        if pretrained:
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            base = resnet18(weights=None)

        # Modify first conv layer for single-channel input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Modified: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Initialize from pretrained weights if available
        if pretrained:
            # Average the pretrained weights across input channels
            pretrained_weight = base.conv1.weight.data
            self.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)

        # Copy remaining layers from ResNet18
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool

        # Embedding projection
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_embed = nn.Linear(feature_dim, embedding_dim)

        # Store dimensions
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            x: Input spectrogram tensor of shape (B, 1, H, W)

        Returns:
            Dictionary containing:
                - 'features': Raw features before embedding (B, feature_dim)
                - 'embedding': Projected embedding (B, embedding_dim)
                - 'intermediate': Intermediate features for adversarial branches
        """
        # Early layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks with intermediate outputs
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Global average pooling
        pooled = self.avgpool(x4)
        features = torch.flatten(pooled, 1)

        # Embedding projection
        features_dropped = self.dropout(features)
        embedding = self.fc_embed(features_dropped)

        return {
            'features': features,
            'embedding': embedding,
            'intermediate': {
                'layer1': x1,
                'layer2': x2,
                'layer3': x3,
                'layer4': x4,
            }
        }

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the embedding output."""
        return self.forward(x)['embedding']

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the raw features."""
        return self.forward(x)['features']

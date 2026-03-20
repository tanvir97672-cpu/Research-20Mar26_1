#!/usr/bin/env python3
"""
Quick verification script to check all imports work.
Run this BEFORE the smoke test to catch any import errors early.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_imports():
    """Verify all module imports work correctly."""
    print("Verifying imports...")
    errors = []

    # Core Python libraries
    try:
        import numpy as np
        print("  [OK] numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")

    try:
        from scipy.ndimage import zoom
        print("  [OK] scipy")
    except ImportError as e:
        errors.append(f"scipy: {e}")

    try:
        import yaml
        print("  [OK] pyyaml")
    except ImportError as e:
        errors.append(f"pyyaml: {e}")

    # PyTorch
    try:
        import torch
        print(f"  [OK] torch (version: {torch.__version__})")
        print(f"       CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"       GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import pytorch_lightning as pl
        print(f"  [OK] pytorch_lightning (version: {pl.__version__})")
    except ImportError as e:
        errors.append(f"pytorch_lightning: {e}")

    try:
        import torchvision
        print(f"  [OK] torchvision (version: {torchvision.__version__})")
    except ImportError as e:
        errors.append(f"torchvision: {e}")

    try:
        import torchmetrics
        print(f"  [OK] torchmetrics (version: {torchmetrics.__version__})")
    except ImportError as e:
        errors.append(f"torchmetrics: {e}")

    try:
        from sklearn.metrics import roc_auc_score
        print("  [OK] scikit-learn")
    except ImportError as e:
        errors.append(f"scikit-learn: {e}")

    # Project modules
    try:
        from data.preprocessing import compute_stft_spectrogram
        print("  [OK] data.preprocessing")
    except ImportError as e:
        errors.append(f"data.preprocessing: {e}")

    try:
        from data.dataset import LoRaRFFIDataset, create_dataloaders
        print("  [OK] data.dataset")
    except ImportError as e:
        errors.append(f"data.dataset: {e}")

    try:
        from models.backbone import SharedBackbone
        print("  [OK] models.backbone")
    except ImportError as e:
        errors.append(f"models.backbone: {e}")

    try:
        from models.evidential import EvidentialHead, evidential_loss
        print("  [OK] models.evidential")
    except ImportError as e:
        errors.append(f"models.evidential: {e}")

    try:
        from models.adversarial import ChannelAdversary, DomainAdversary
        print("  [OK] models.adversarial")
    except ImportError as e:
        errors.append(f"models.adversarial: {e}")

    try:
        from models.daos_rff import DAOS_RFF
        print("  [OK] models.daos_rff")
    except ImportError as e:
        errors.append(f"models.daos_rff: {e}")

    try:
        from utils.metrics import compute_open_set_metrics
        print("  [OK] utils.metrics")
    except ImportError as e:
        errors.append(f"utils.metrics: {e}")

    try:
        from utils.losses import supervised_contrastive_loss
        print("  [OK] utils.losses")
    except ImportError as e:
        errors.append(f"utils.losses: {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print("IMPORT VERIFICATION FAILED!")
        print("=" * 50)
        for err in errors:
            print(f"  ERROR: {err}")
        return False
    else:
        print("ALL IMPORTS VERIFIED SUCCESSFULLY!")
        print("=" * 50)
        return True


def verify_model_creation():
    """Quick test of model instantiation."""
    print("\nVerifying model creation...")

    try:
        from models.daos_rff import DAOS_RFF
        model = DAOS_RFF(
            pretrained=False,
            num_classes=5,
            num_channels=3,
            num_domains=2,
        )
        print("  [OK] DAOS_RFF model created")

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  [OK] Model has {params:,} parameters")

        # Test forward pass with dummy input
        import torch
        dummy_input = torch.randn(2, 1, 129, 64)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  [OK] Forward pass successful")
        print(f"       Output keys: {list(output.keys())}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_imports()
    if success:
        success = verify_model_creation()

    sys.exit(0 if success else 1)

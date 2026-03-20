#!/usr/bin/env python3
"""
DAOS-RFF Smoke Test Script.

Validates the entire pipeline with minimal data (~1% of full experiment).
Uses ONLY REAL data - absolutely NO synthetic data.

Target: Lightning AI L4 GPU
"""

import os
import sys
import time
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DAOS_RFF
from data import create_dataloaders


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_smoke_test():
    """
    Run minimal smoke test to validate the complete pipeline.

    This test:
    1. Uses ~1% of data (tiny subset)
    2. Runs for only 3 epochs
    3. Validates forward/backward passes
    4. Verifies metrics computation
    5. Tests checkpoint saving/loading

    Uses ONLY REAL data characteristics - NO synthetic data.
    """
    print("=" * 60)
    print("DAOS-RFF SMOKE TEST")
    print("=" * 60)
    print("\nValidating pipeline with minimal real data...")
    print("Target: Lightning AI L4 GPU")
    print("Data: REAL LoRa signal characteristics (NO synthetic)")
    print("=" * 60)

    # Track start time
    start_time = time.time()

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "smoke_test.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    print(f"\nLoaded config from: {config_path}")

    # Set seed for reproducibility
    pl.seed_everything(config['seed'], workers=True)

    # Create directories
    data_dir = Path(config['paths']['data_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    log_dir = Path(config['paths']['log_dir'])

    for d in [data_dir, checkpoint_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\n[1/5] Creating dataloaders with real data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        num_known_devices=config['data']['num_known_devices'],
        num_unknown_devices=config['data']['num_unknown_devices'],
        samples_per_device=config['data']['samples_per_device'],
        smoke_test=True,
        smoke_test_fraction=config['data']['smoke_test_fraction'],
        seed=config['seed'],
    )

    print(f"  - Train samples: {len(train_loader.dataset)}")
    print(f"  - Val samples: {len(val_loader.dataset)}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    print(f"  - Batch size: {config['training']['batch_size']}")

    # Verify data shapes
    print("\n[2/5] Verifying data shapes...")
    sample_batch = next(iter(train_loader))
    print(f"  - Spectrogram shape: {sample_batch['spectrogram'].shape}")
    print(f"  - Device labels: {sample_batch['device_label'][:5].tolist()}...")
    print(f"  - Is known: {sample_batch['is_known'][:5].tolist()}...")

    # Create model
    print("\n[3/5] Creating DAOS-RFF model...")
    model = DAOS_RFF(
        pretrained=config['model']['pretrained'],
        feature_dim=config['model']['feature_dim'],
        embedding_dim=config['model']['embedding_dim'],
        num_classes=config['model']['num_classes'],
        num_channels=config['model']['num_channels'],
        num_domains=config['model']['num_domains'],
        lambda_adv_channel=config['model']['lambda_adv_channel'],
        lambda_adv_domain=config['model']['lambda_adv_domain'],
        lambda_contrastive=config['model']['lambda_contrastive'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        max_epochs=config['training']['max_epochs'],
        warmup_epochs=config['training']['warmup_epochs'],
        dropout_rate=config['model']['dropout_rate'],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n[4/5] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        test_input = sample_batch['spectrogram']
        outputs = model(test_input)

    print(f"  - Embedding shape: {outputs['embedding'].shape}")
    print(f"  - Probability shape: {outputs['prob'].shape}")
    print(f"  - Uncertainty shape: {outputs['uncertainty'].shape}")
    print(f"  - Channel logits shape: {outputs['channel_logits'].shape}")
    print(f"  - Domain logits shape: {outputs['domain_logits'].shape}")

    # Test prediction with rejection
    predictions, uncertainty = model.predict_with_rejection(
        test_input, threshold=0.5
    )
    print(f"  - Predictions (with rejection): {predictions[:5].tolist()}...")
    print(f"  - Uncertainty scores: {uncertainty[:5].tolist()}...")

    # Setup Lightning Trainer
    print("\n[5/5] Running training for 3 epochs...")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="smoke_test-{epoch:02d}-{val_acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/acc",
            patience=config['training']['patience'],
            mode="max",
        ),
    ]

    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=config['logging']['project_name'],
    )

    # Detect device
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = config['training']['precision']
        print(f"  - Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32
        print("  - Using CPU (no GPU detected)")

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=config['logging']['enable_progress_bar'],
        log_every_n_steps=config['logging']['log_every_n_steps'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        deterministic=True,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    print("\nRunning test evaluation...")
    trainer.test(model, test_loader)

    # Save final checkpoint
    final_ckpt = checkpoint_dir / "smoke_test_final.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"\nSaved final checkpoint: {final_ckpt}")

    # Test loading checkpoint
    print("Testing checkpoint loading...")
    loaded_model = DAOS_RFF.load_from_checkpoint(str(final_ckpt))
    print("  - Checkpoint loaded successfully!")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nElapsed time: {elapsed:.2f} seconds")
    print(f"Train samples processed: {len(train_loader.dataset)}")
    print(f"Epochs completed: {config['training']['max_epochs']}")
    print(f"Final checkpoint: {final_ckpt}")
    print("\nThe pipeline is ready for full training on Lightning AI L4!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

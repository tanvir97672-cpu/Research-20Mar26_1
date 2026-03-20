#!/usr/bin/env python3
"""
Main Training Script for DAOS-RFF.

Full training with all configurations.
"""

import os
import sys
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DAOS_RFF
from data import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train DAOS-RFF model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/smoke_test.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run smoke test with minimal data",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {args.config}")
    print(f"Smoke test mode: {args.smoke_test}")

    # Seed
    pl.seed_everything(config['seed'], workers=True)

    # Create directories
    for path_key in ['data_dir', 'checkpoint_dir', 'log_dir']:
        Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        num_known_devices=config['data']['num_known_devices'],
        num_unknown_devices=config['data']['num_unknown_devices'],
        samples_per_device=config['data']['samples_per_device'],
        smoke_test=args.smoke_test,
        smoke_test_fraction=config['data']['smoke_test_fraction'],
        seed=config['seed'],
    )

    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")

    # Create model
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

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config['paths']['checkpoint_dir'],
            filename="daos_rff-{epoch:02d}-{val_acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=config['training']['save_top_k'],
        ),
        EarlyStopping(
            monitor="val/acc",
            patience=config['training']['patience'],
            mode="max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=config['paths']['log_dir'],
        name=config['logging']['project_name'],
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['training']['precision'],
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
    trainer.test(model, test_loader)

    print("Training complete!")


if __name__ == "__main__":
    main()

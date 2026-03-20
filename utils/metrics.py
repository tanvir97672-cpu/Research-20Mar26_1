"""
Metrics for Open-Set RF Fingerprinting Evaluation.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from typing import Dict, Tuple


def compute_open_set_metrics(
    uncertainty: np.ndarray,
    is_known: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute comprehensive open-set detection metrics.

    Args:
        uncertainty: Uncertainty scores (N,)
        is_known: Boolean array, True for known samples (N,)
        predictions: Predicted class labels (N,)
        true_labels: True class labels (N,), -1 for unknown
        num_classes: Number of known classes

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Binary labels for open-set detection: 1 = unknown, 0 = known
    is_unknown = ~is_known

    # 1. AUROC for open-set detection
    if is_unknown.sum() > 0 and is_known.sum() > 0:
        metrics['auroc'] = roc_auc_score(is_unknown.astype(int), uncertainty)
    else:
        metrics['auroc'] = 0.0

    # 2. AUPR (Area Under Precision-Recall Curve)
    if is_unknown.sum() > 0 and is_known.sum() > 0:
        precision, recall, _ = precision_recall_curve(
            is_unknown.astype(int), uncertainty
        )
        metrics['aupr'] = auc(recall, precision)
    else:
        metrics['aupr'] = 0.0

    # 3. FPR at 95% TPR
    if is_unknown.sum() > 0 and is_known.sum() > 0:
        fpr, tpr, _ = roc_curve(is_unknown.astype(int), uncertainty)
        # Find FPR when TPR >= 0.95
        idx = np.argmax(tpr >= 0.95)
        metrics['fpr_at_95_tpr'] = fpr[idx] if tpr[idx] >= 0.95 else 1.0
    else:
        metrics['fpr_at_95_tpr'] = 1.0

    # 4. Closed-set accuracy (only on known samples)
    known_mask = is_known
    if known_mask.sum() > 0:
        known_preds = predictions[known_mask]
        known_true = true_labels[known_mask]
        metrics['closed_set_acc'] = (known_preds == known_true).mean()
    else:
        metrics['closed_set_acc'] = 0.0

    # 5. Open-set classification rate (OSCR-style)
    # Correctly classified known + correctly rejected unknown
    if is_known.sum() > 0 and is_unknown.sum() > 0:
        # Find optimal threshold (could also be fixed)
        threshold = np.median(uncertainty)

        # Known samples correctly classified (below threshold and correct)
        known_correct = (
            known_mask &
            (uncertainty <= threshold) &
            (predictions == true_labels)
        ).sum()

        # Unknown samples correctly rejected (above threshold)
        unknown_rejected = (is_unknown & (uncertainty > threshold)).sum()

        total = len(uncertainty)
        metrics['oscr'] = (known_correct + unknown_rejected) / total
    else:
        metrics['oscr'] = 0.0

    return metrics


def find_optimal_threshold(
    uncertainty: np.ndarray,
    is_known: np.ndarray,
    metric: str = 'youden'
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal uncertainty threshold for open-set detection.

    Args:
        uncertainty: Uncertainty scores
        is_known: Boolean mask for known samples
        metric: Optimization metric ('youden', 'f1', 'eer')

    Returns:
        Optimal threshold and metrics at that threshold
    """
    is_unknown = ~is_known

    if is_unknown.sum() == 0 or is_known.sum() == 0:
        return 0.5, {}

    fpr, tpr, thresholds = roc_curve(is_unknown.astype(int), uncertainty)

    if metric == 'youden':
        # Youden's J statistic: TPR - FPR
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)

    elif metric == 'eer':
        # Equal Error Rate: where FPR = FNR
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        optimal_idx = eer_idx

    else:
        # Default to median threshold
        optimal_idx = len(thresholds) // 2

    optimal_threshold = thresholds[optimal_idx]

    metrics = {
        'threshold': optimal_threshold,
        'tpr': tpr[optimal_idx],
        'fpr': fpr[optimal_idx],
        'eer': (fpr[optimal_idx] + (1 - tpr[optimal_idx])) / 2,
    }

    return optimal_threshold, metrics

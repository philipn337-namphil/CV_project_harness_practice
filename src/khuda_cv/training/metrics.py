"""Canonical training metrics for KHUDA CV."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class EpochMetrics:
    loss: float
    clip_accuracy: float
    clip_f1: float
    frame_f1: float


def compute_epoch_metrics(
    average_loss,
    clip_preds,
    clip_labels,
    frame_preds,
    frame_labels,
):
    """Compute canonical epoch-level metrics."""

    clip_preds = np.asarray(clip_preds).reshape(-1)
    clip_labels = np.asarray(clip_labels).reshape(-1)

    frame_preds = np.asarray(frame_preds).reshape(-1)
    frame_labels = np.asarray(frame_labels).reshape(-1)

    clip_accuracy = float(
        (clip_preds == clip_labels).mean()
    )

    clip_f1 = float(
        f1_score(
            clip_labels,
            clip_preds,
            average="macro",
            zero_division=0,
        )
    )

    frame_f1 = float(
        f1_score(
            frame_labels,
            frame_preds,
            average="binary",
            zero_division=0,
        )
    )

    return EpochMetrics(
        loss=float(average_loss),
        clip_accuracy=clip_accuracy,
        clip_f1=clip_f1,
        frame_f1=frame_f1,
    )
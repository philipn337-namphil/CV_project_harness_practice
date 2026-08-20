"""Canonical training losses for KHUDA CV."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.khuda_cv.model import FocalLoss


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    clip_loss: torch.Tensor
    frame_loss: torch.Tensor


def build_losses(class_counts=None, focal_gamma=2.0):
    """Build the canonical clip-level and frame-level loss functions."""

    alpha = None

    if class_counts:
        total = sum(class_counts.values())
        alpha = torch.tensor(
            [
                total / (2.0 * class_counts.get(0, 1)),
                total / (2.0 * class_counts.get(1, 1)),
            ],
            dtype=torch.float32,
        )

    clip_criterion = FocalLoss(
        gamma=focal_gamma,
        alpha=alpha,
    )

    frame_criterion = nn.BCEWithLogitsLoss()

    return clip_criterion, frame_criterion


def compute_losses(
    outputs,
    batch,
    clip_criterion,
    frame_criterion,
):
    """Compute losses from the canonical model output and batch contracts."""

    clip_logits = outputs["clip_logits"]
    frame_logits = outputs["frame_logits"]

    clip_labels = batch["clip_label"]
    frame_labels = batch["frame_labels"]

    clip_loss = clip_criterion(
        clip_logits,
        clip_labels,
    )

    frame_loss = frame_criterion(
        frame_logits,
        frame_labels,
    )

    total_loss = clip_loss + frame_loss

    return LossOutput(
        total_loss=total_loss,
        clip_loss=clip_loss,
        frame_loss=frame_loss,
    )
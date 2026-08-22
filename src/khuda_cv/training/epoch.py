"""Canonical epoch-level training and validation logic for KHUDA CV."""

import torch

from .losses import compute_losses
from .metrics import compute_epoch_metrics
from .step import move_batch_to_device, train_step


def train_epoch(
    model,
    loader,
    optimizer,
    clip_criterion,
    frame_criterion,
    device="cpu",
):
    """Run one canonical training epoch."""

    model.train()

    running_loss = 0.0

    all_clip_preds = []
    all_clip_labels = []

    all_frame_preds = []
    all_frame_labels = []

    num_batches = 0

    for batch in loader:
        result = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            clip_criterion=clip_criterion,
            frame_criterion=frame_criterion,
            device=device,
        )

        running_loss += result["total_loss"].item()
        num_batches += 1

        clip_preds = result["clip_logits"].argmax(dim=1)
        frame_preds = (result["frame_logits"] > 0).long()

        all_clip_preds.extend(
            clip_preds.cpu().tolist()
        )

        all_clip_labels.extend(
            batch["clip_label"].cpu().tolist()
        )

        all_frame_preds.extend(
            frame_preds.cpu().tolist()
        )

        all_frame_labels.extend(
            batch["frame_labels"].long().cpu().tolist()
        )

    if num_batches == 0:
        raise ValueError("train_epoch received an empty loader.")

    average_loss = running_loss / num_batches

    return compute_epoch_metrics(
        average_loss=average_loss,
        clip_preds=all_clip_preds,
        clip_labels=all_clip_labels,
        frame_preds=all_frame_preds,
        frame_labels=all_frame_labels,
    )


@torch.no_grad()
def validate_epoch(
    model,
    loader,
    clip_criterion,
    frame_criterion,
    device="cpu",
):
    """Run one canonical validation epoch."""

    model.eval()

    running_loss = 0.0

    all_clip_preds = []
    all_clip_labels = []

    all_frame_preds = []
    all_frame_labels = []

    num_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        outputs = model(batch["pixel_values"])

        losses = compute_losses(
            outputs,
            batch,
            clip_criterion,
            frame_criterion,
        )

        running_loss += losses.total_loss.item()
        num_batches += 1

        clip_preds = outputs["clip_logits"].argmax(dim=1)
        frame_preds = (outputs["frame_logits"] > 0).long()

        all_clip_preds.extend(
            clip_preds.cpu().tolist()
        )

        all_clip_labels.extend(
            batch["clip_label"].cpu().tolist()
        )

        all_frame_preds.extend(
            frame_preds.cpu().tolist()
        )

        all_frame_labels.extend(
            batch["frame_labels"].long().cpu().tolist()
        )

    if num_batches == 0:
        raise ValueError("validate_epoch received an empty loader.")

    average_loss = running_loss / num_batches

    return compute_epoch_metrics(
        average_loss=average_loss,
        clip_preds=all_clip_preds,
        clip_labels=all_clip_labels,
        frame_preds=all_frame_preds,
        frame_labels=all_frame_labels,
    )
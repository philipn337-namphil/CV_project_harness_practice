"""Canonical single-step training logic for KHUDA CV."""

import torch

from .losses import compute_losses


def move_batch_to_device(batch, device):
    """Move tensor fields in the canonical batch to the target device."""

    return {
        "pixel_values": batch["pixel_values"].to(device),
        "clip_label": batch["clip_label"].to(device),
        "frame_labels": batch["frame_labels"].to(device),
        "clip_id": batch.get("clip_id"),
        "video_path": batch.get("video_path"),
    }


def train_step(
    model,
    batch,
    optimizer,
    clip_criterion,
    frame_criterion,
    device="cpu",
):
    """Run one canonical optimization step."""

    model.train()

    batch = move_batch_to_device(batch, device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(batch["pixel_values"])

    losses = compute_losses(
        outputs,
        batch,
        clip_criterion,
        frame_criterion,
    )

    losses.total_loss.backward()
    optimizer.step()

    return {
        "total_loss": losses.total_loss.detach(),
        "clip_loss": losses.clip_loss.detach(),
        "frame_loss": losses.frame_loss.detach(),
        "clip_logits": outputs["clip_logits"].detach(),
        "frame_logits": outputs["frame_logits"].detach(),
    }
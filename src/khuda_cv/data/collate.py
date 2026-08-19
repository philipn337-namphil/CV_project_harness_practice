"""Canonical batch collation for the KHUDA CV data pipeline."""

import torch


def collate_fn(batch):
    """Collate dictionary samples into the canonical batch contract."""

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    clip_labels = torch.stack([item["clip_label"] for item in batch])
    frame_labels = torch.stack([item["frame_labels"] for item in batch])

    clip_ids = [item["clip_id"] for item in batch]
    video_paths = [item["video_path"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "clip_label": clip_labels,
        "frame_labels": frame_labels,
        "clip_id": clip_ids,
        "video_path": video_paths,
    }
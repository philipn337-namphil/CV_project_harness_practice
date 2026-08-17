import torch


def make_sample():
    return {
        "pixel_values": torch.randn(16, 3, 224, 224),
        "clip_label": torch.tensor(1, dtype=torch.long),
        "frame_labels": torch.zeros(16, dtype=torch.float32),
        "clip_id": "sample_clip_001",
        "video_path": "sample_video.mp4",
    }


def canonical_collate(batch):
    return {
        "pixel_values": torch.stack(
            [item["pixel_values"] for item in batch]
        ),
        "clip_label": torch.stack(
            [item["clip_label"] for item in batch]
        ),
        "frame_labels": torch.stack(
            [item["frame_labels"] for item in batch]
        ),
        "clip_id": [
            item["clip_id"] for item in batch
        ],
        "video_path": [
            item["video_path"] for item in batch
        ],
    }


def test_dataset_sample_contract():
    sample = make_sample()

    assert isinstance(sample, dict)

    required_keys = {
        "pixel_values",
        "clip_label",
        "frame_labels",
        "clip_id",
        "video_path",
    }

    assert required_keys.issubset(sample.keys())

    assert sample["pixel_values"].shape == (16, 3, 224, 224)
    assert sample["clip_label"].ndim == 0
    assert sample["frame_labels"].shape == (16,)

    assert sample["clip_label"].item() in (0, 1)


def test_dataloader_batch_contract():
    batch = canonical_collate([
        make_sample(),
        make_sample(),
    ])

    assert isinstance(batch, dict)

    assert batch["pixel_values"].shape == (2, 16, 3, 224, 224)
    assert batch["clip_label"].shape == (2,)
    assert batch["frame_labels"].shape == (2, 16)

    assert len(batch["clip_id"]) == 2
    assert len(batch["video_path"]) == 2
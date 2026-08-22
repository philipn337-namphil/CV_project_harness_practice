import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.khuda_cv.training import (
    build_losses,
    train_epoch,
    validate_epoch,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_head = nn.Linear(4, 2)
        self.frame_head = nn.Linear(4, 16)

    def forward(self, pixel_values):
        x = pixel_values.mean(dim=(1, 2, 3, 4))
        x = x.unsqueeze(1).repeat(1, 4)

        return {
            "clip_logits": self.clip_head(x),
            "frame_logits": self.frame_head(x),
        }


def _make_sample(index):
    return {
        "pixel_values": torch.randn(16, 3, 8, 8),
        "clip_label": torch.tensor(index % 2, dtype=torch.long),
        "frame_labels": torch.tensor(
            [index % 2] * 16,
            dtype=torch.float32,
        ),
        "clip_id": f"clip-{index}",
        "video_path": f"video-{index}.mp4",
    }


def _make_loader():
    samples = [_make_sample(i) for i in range(4)]

    return DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
    )


def test_train_epoch_contract():
    torch.manual_seed(42)

    model = ToyModel()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    clip_criterion, frame_criterion = build_losses()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = train_epoch(
        model=model,
        loader=_make_loader(),
        optimizer=optimizer,
        clip_criterion=clip_criterion,
        frame_criterion=frame_criterion,
        device="cpu",
    )

    after = list(model.parameters())

    assert model.training is True

    assert torch.isfinite(torch.tensor(metrics.loss))

    assert 0.0 <= metrics.clip_accuracy <= 1.0
    assert 0.0 <= metrics.clip_f1 <= 1.0
    assert 0.0 <= metrics.frame_f1 <= 1.0

    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, after)
    )


def test_validate_epoch_contract():
    torch.manual_seed(42)

    model = ToyModel()

    clip_criterion, frame_criterion = build_losses()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = validate_epoch(
        model=model,
        loader=_make_loader(),
        clip_criterion=clip_criterion,
        frame_criterion=frame_criterion,
        device="cpu",
    )

    after = list(model.parameters())

    assert model.training is False

    assert torch.isfinite(torch.tensor(metrics.loss))

    assert 0.0 <= metrics.clip_accuracy <= 1.0
    assert 0.0 <= metrics.clip_f1 <= 1.0
    assert 0.0 <= metrics.frame_f1 <= 1.0

    assert all(
        torch.equal(old, new.detach())
        for old, new in zip(before, after)
    )
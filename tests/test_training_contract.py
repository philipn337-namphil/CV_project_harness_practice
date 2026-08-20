import torch
import torch.nn as nn

from src.khuda_cv.training import build_losses, compute_losses, train_step


class ToyCanonicalModel(nn.Module):
    """Small model that follows the canonical model output contract."""

    def __init__(self):
        super().__init__()

        self.clip_head = nn.Linear(4, 2)
        self.frame_head = nn.Linear(4, 16)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]

        features = pixel_values.reshape(batch_size, -1)[:, :4]

        return {
            "clip_logits": self.clip_head(features),
            "frame_logits": self.frame_head(features),
        }


def make_batch(batch_size=2):
    return {
        "pixel_values": torch.randn(batch_size, 16, 3, 4, 4),
        "clip_label": torch.tensor([0, 1], dtype=torch.long),
        "frame_labels": torch.zeros(batch_size, 16),
        "clip_id": ["clip_0", "clip_1"],
        "video_path": ["video_0.mp4", "video_1.mp4"],
    }


def test_training_loss_contract():
    clip_criterion, frame_criterion = build_losses()

    outputs = {
        "clip_logits": torch.randn(2, 2),
        "frame_logits": torch.randn(2, 16),
    }

    batch = make_batch()

    losses = compute_losses(
        outputs,
        batch,
        clip_criterion,
        frame_criterion,
    )

    assert losses.total_loss.ndim == 0
    assert losses.clip_loss.ndim == 0
    assert losses.frame_loss.ndim == 0

    assert torch.isfinite(losses.total_loss)
    assert torch.isfinite(losses.clip_loss)
    assert torch.isfinite(losses.frame_loss)

    assert torch.allclose(
        losses.total_loss,
        losses.clip_loss + losses.frame_loss,
    )


def test_single_train_step_contract():
    torch.manual_seed(42)

    model = ToyCanonicalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    clip_criterion, frame_criterion = build_losses()

    batch = make_batch()

    before = model.clip_head.weight.detach().clone()

    result = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        clip_criterion=clip_criterion,
        frame_criterion=frame_criterion,
        device="cpu",
    )

    after = model.clip_head.weight.detach().clone()

    assert result["total_loss"].ndim == 0
    assert result["clip_loss"].ndim == 0
    assert result["frame_loss"].ndim == 0

    assert result["clip_logits"].shape == (2, 2)
    assert result["frame_logits"].shape == (2, 16)

    assert torch.isfinite(result["total_loss"])

    assert not torch.equal(before, after)
from pathlib import Path
import importlib.util

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "src" / "khuda_cv" / "model.py"

CHECKPOINT_PATH = (
    Path.home()
    / "Desktop"
    / "project_package"
    / "checkpoints"
    / "best_model.pt"
)


def load_model_module():
    spec = importlib.util.spec_from_file_location("canonical_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="best_model.pt is not available locally",
)
def test_checkpoint_contract():
    module = load_model_module()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    assert isinstance(checkpoint, dict)
    assert "model_state_dict" in checkpoint

    state_dict = checkpoint["model_state_dict"]

    required_keys = {
        "clip_head.0.weight",
        "clip_head.0.bias",
        "clip_head.3.weight",
        "clip_head.3.bias",
        "frame_head.0.weight",
        "frame_head.0.bias",
        "frame_head.3.weight",
        "frame_head.3.bias",
    }

    assert required_keys.issubset(state_dict.keys())

    backbone_keys = [
        key for key in state_dict
        if key.startswith("backbone.")
    ]

    assert len(backbone_keys) == 182

    model = module.GarbageDumpingVideoMAE()

    # strict=True가 핵심
    model.load_state_dict(state_dict, strict=True)
from pathlib import Path
import importlib.util

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "legacy" / "final_package" / "model.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("canonical_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


def test_model_output_contract():
    module = load_model_module()

    model = module.GarbageDumpingVideoMAE(
        pretrained_name="MCG-NJU/videomae-base-finetuned-kinetics"
    )
    model.eval()

    dummy = torch.randn(1, 16, 3, 224, 224)

    with torch.no_grad():
        outputs = model(dummy)

    assert isinstance(outputs, dict)

    assert "clip_logits" in outputs
    assert "frame_logits" in outputs
    assert "last_hidden_state" in outputs

    assert outputs["clip_logits"].shape == (1, 2)
    assert outputs["frame_logits"].shape == (1, 16)
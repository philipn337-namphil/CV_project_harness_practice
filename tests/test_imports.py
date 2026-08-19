from pathlib import Path
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "src" / "khuda_cv" / "model.py"


def test_canonical_model_file_exists():
    assert MODEL_PATH.exists()


def test_canonical_model_import():
    spec = importlib.util.spec_from_file_location("canonical_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert hasattr(module, "GarbageDumpingVideoMAE")
    assert hasattr(module, "build_model")
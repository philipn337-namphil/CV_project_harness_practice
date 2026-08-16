# Execution Baseline

This document records the Day 2 local execution baseline.

## Observed Environment

| Component | Version |
| --- | --- |
| Python | `3.12.10` |
| torch | `2.11.0+cu128` |
| transformers | `5.4.0` |
| OpenCV | `4.13.0` |
| decord | initially missing |

The final-package requirements list:

```text
torch
transformers
decord
ultralytics
opencv-python
numpy
scipy
```

## Known Blockers

- `decord` was initially missing in the local environment, so dataset imports can fail before any training logic runs.
- The repo has hard-coded Linux data/checkpoint paths such as `/data/leecg1219/...`, `/data/philipn337/...`, and `/data2/local_datasets/...`.
- The current `seraph/train.py` path uses `VideoMAEMultiHead`, tuple batches, and one-logit BCE classification.
- The canonical final-package path uses `GarbageDumpingVideoMAE`, dict outputs, and two-class `clip_logits`.
- Local Windows execution may need path overrides before dataset or pipeline scripts can run.

## Smoke Checks

Use these checks before production refactors.

### Repository State

```powershell
git status
git diff --stat
```

### Import Checks

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import decord; print(decord.__version__)"
```

`decord` may fail until installed in the active environment. Record that as a dependency blocker, not as a model-contract failure.

### Canonical Model Shape Check

Run from the package location that can import `GarbageDumpingVideoMAE`:

```powershell
python -c "import torch; from model import GarbageDumpingVideoMAE; m=GarbageDumpingVideoMAE(pretrained_name='MCG-NJU/videomae-base'); x=torch.randn(1,16,3,224,224); y=m(x); print(y['clip_logits'].shape, y['frame_logits'].shape)"
```

Expected:

```text
torch.Size([1, 2]) torch.Size([1, 16])
```

This check may download Hugging Face weights if they are not cached.

### Dataset Contract Check

Once `decord` and local data paths are available, verify:

```powershell
python -c "from seraph.dataset import GarbageDumpingClipDataset, collate_fn; print('dataset import ok')"
```

Then test one manifest sample and one collated batch. Expected batch keys:

```text
pixel_values, clip_label, frame_labels, clip_id
```

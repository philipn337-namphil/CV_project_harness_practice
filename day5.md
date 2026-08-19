# Day 5 - Canonicalize Data Pipeline

## 1. Day 4 Baseline

- Repository: `C:\Users\phili\CV_project_harness_practice`
- Started from the latest `main` after repository sync.
- Confirmed the working tree was clean before Day 5 changes.
- Ran the full harness with `python -m pytest`.
- Baseline result: `9 passed`.

## 2. Dataset Implementation Comparison

- Compared `seraph/dataset.py` and `seraph/other/dataset.py`.
- Both files were `8276 bytes`.
- Both files had the same SHA256:

```text
68CDBAFA3AD6B5CA6FE820943D62B4228C6E508EDEB9C9562B9BED43782A0C93
```

- Both contained the same `GarbageDumpingClipDataset` and `collate_fn`.
- Reviewed `seraph/utils/transforms.py`.
- Found `_FolderClipDataset`, `MyDataset`, and `_tuple_collate_fn`.
- Also found `from dataset import GarbageDumpingClipDataset`.
- Identified a possible interface mismatch between manifest-based dictionary samples and tuple-based collation.

## 3. Canonical Dataset Contract

The canonical sample contract is dictionary-based:

```text
pixel_values: (16, 3, 224, 224)
clip_label: scalar
frame_labels: (16,)
clip_id: metadata
video_path: metadata
```

The canonical batch contract is also dictionary-based:

```text
pixel_values: (B, 16, 3, 224, 224)
clip_label: (B,)
frame_labels: (B, 16)
clip_id: list
video_path: list
```

Decision:

- Canonical code uses the manifest-based dictionary contract.
- Existing `seraph` data code remains as legacy reference.

## 4. Canonical Data Structure

Created the canonical data package:

```text
src/khuda_cv/data/__init__.py
src/khuda_cv/data/dataset.py
src/khuda_cv/data/collate.py
```

## 5. Dataset Code Migration

- Used `seraph/dataset.py` as the base for the canonical dataset implementation.
- Removed the legacy `if __name__ == "__main__"` execution block.
- Preserved `GarbageDumpingClipDataset` as the canonical dataset class.
- Encountered `ModuleNotFoundError` because `decord` was not installed.
- Installed `decord 0.6.0`.
- After that, Windows produced one `c10.dll` `WinError 1114` issue when `decord` loaded before `torch`.
- Verified `torch 2.11.0+cu128` imported successfully by itself.
- Fixed the canonical import order so `torch` is imported before `decord`.
- Confirmed canonical dataset import with `canonical dataset import ok`.

## 6. DataLoader And Collate

Implemented canonical dictionary collation in `src/khuda_cv/data/collate.py`.

The canonical `collate_fn`:

- Stacks `pixel_values`.
- Stacks `clip_label`.
- Stacks `frame_labels`.
- Preserves `clip_id` as metadata list.
- Preserves `video_path` as metadata list.

Exported both public data APIs from `src/khuda_cv/data/__init__.py`:

```python
from .collate import collate_fn
from .dataset import GarbageDumpingClipDataset
```

Synthetic smoke test result:

```text
pixel_values: torch.Size([2, 16, 3, 224, 224])
clip_label: torch.Size([2])
frame_labels: torch.Size([2, 16])
clip_id: metadata list ok
video_path: metadata list ok
```

This also fixes the canonical path so `video_path` is preserved in the batch contract, unlike the old legacy tuple collation path where it could be lost.

## 7. Dataset Harness Import Path

Updated `tests/test_dataset_contract.py`.

- Removed the synthetic `canonical_collate` helper from the test.
- Switched the test to import the real canonical collate function:

```python
from src.khuda_cv.data import collate_fn
```

Focused dataset contract result:

```text
2 passed
```

## 8. Focused Contract Verification

Ran the focused contract tests for dataset and preprocessing.

Result:

```text
Dataset contract: 2 passed
Preprocessing contract: 3 passed
Total: 5 passed
```

## 9. Full Harness Verification

Ran the full harness:

```powershell
python -m pytest
```

Final result:

```text
9 passed in 18.88s
```

No regression was introduced by the canonical data pipeline work.

## 10. Day 5 Result

Day 5 established the canonical data path:

```text
Dataset -> Collate -> Batch
```

The project now has a canonical dictionary-based data pipeline under `src/khuda_cv/data/`.

Summary of the learning flow so far:

- Day 1: Observe the existing project behavior.
- Day 2: Define the expected contracts.
- Day 3: Enforce those contracts through tests.
- Day 4: Canonicalize the model path.
- Day 5: Canonicalize the data pipeline.

Legacy data code remains available as historical reference, while new code should use the canonical `src/khuda_cv/data` package.

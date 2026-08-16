# Architecture Contract

This document records the Day 2 architecture decision for the harness practice repository.

## System Overview

The project is organized around a garbage dumping video detection workflow:

```text
KHUDA_173 manifest/data
        |
        v
Dataset and frame sampling
        |
        v
VideoMAE input tensor
        |
        v
GarbageDumpingVideoMAE
        |
        +--> clip-level legal/illegal classification
        |
        +--> frame-level illegal-event logits
        |
        v
Pipeline/evaluation/visualization outputs
```

The canonical runtime contract is now based on `GarbageDumpingVideoMAE`, copied into `legacy/final_package/model.py` from the final package that produced the verified checkpoint. The current `seraph/` code remains part of the repository history and training exploration, but it is not the canonical interface for future harness work.

## Canonical Model

`GarbageDumpingVideoMAE` is the canonical model for Day 2 and later harness tasks.

Contract:

- Input: `pixel_values`
- Input shape: `(B, 16, 3, 224, 224)`
- Backbone: Hugging Face `VideoMAEModel`
- Output type: dict
- Output keys:
  - `clip_logits`: `(B, 2)`
  - `frame_logits`: `(B, 16)`
  - `last_hidden_state`

`clip_logits` uses two classes:

- `0 = legal`
- `1 = illegal`

`frame_logits` maps model-time positions back to the 16 sampled input frames.

## Legacy Training Path

`VideoMAEMultiHead` in `seraph/models/model.py` is a legacy training path. It differs from the canonical model in important ways:

- It returns a tuple, not a dict.
- Its clip head currently produces one binary logit.
- Its temporal output length follows VideoMAE tubelet grouping from the chosen `num_frames`.
- `seraph/train.py` uses tuple-style batches and BCE-style clip loss.

This code can be studied or refactored later, but future harness tests should target the canonical contract first.

## Canonical vs Legacy Relationship

`legacy/final_package/` is protected reference material. The name `legacy` does not mean obsolete in this context; it means "preserved external final package." Its `GarbageDumpingVideoMAE` implementation is currently the compatibility target for `best_model.pt`.

`seraph/` is the existing project implementation. It contains useful dataset, training, evaluation, and utility code, but it includes known mismatches that must be repaired carefully after contract tests exist.

Recommended future direction:

1. Add harness tests for canonical model shapes and checkpoint keys.
2. Add dataset/batch contract tests for dict batches.
3. Refactor `seraph/` toward the canonical model and batch interface in small commits.

## Protected Architecture Decisions

- Keep label meaning fixed: `0 = legal`, `1 = illegal`.
- Keep VideoMAE input fixed at 16 sampled frames unless a migration is explicitly planned.
- Keep source clip/window length separate from model input length.
- Keep the canonical output dict stable.
- Preserve checkpoint compatibility with `GarbageDumpingVideoMAE`.

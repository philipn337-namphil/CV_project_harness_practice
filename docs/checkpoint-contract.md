# Checkpoint Contract

This document records the verified checkpoint compatibility target.

## Canonical Checkpoint

`best_model.pt` is the protected checkpoint target for the Day 2 contract.

The checkpoint has been verified to contain a top-level `model_state_dict`. That state dict is compatible with `GarbageDumpingVideoMAE`.

Verified key structure:

- 182 keys with prefix `backbone.*`
- `clip_head.0.weight`
- `clip_head.0.bias`
- `clip_head.3.weight`
- `clip_head.3.bias`
- `frame_head.0.weight`
- `frame_head.0.bias`
- `frame_head.3.weight`
- `frame_head.3.bias`

This means the checkpoint expects the canonical MLP heads:

```text
clip_head:
  Linear(hidden, hidden // 2)
  GELU
  Dropout
  Linear(hidden // 2, 2)

frame_head:
  Linear(hidden, hidden // 2)
  GELU
  Dropout
  Linear(hidden // 2, 1)
```

## Compatible Model

The compatible model class is:

```text
GarbageDumpingVideoMAE
```

Expected forward contract:

```python
outputs = model(pixel_values)
outputs["clip_logits"]       # (B, 2)
outputs["frame_logits"]      # (B, 16)
outputs["last_hidden_state"]
```

## Incompatible Legacy Path

`VideoMAEMultiHead` is not the canonical checkpoint target. It has different head names and output structure, so loading `best_model.pt` into it should not be assumed to work.

## Prohibited Changes Without Migration

Do not make these changes unless the task explicitly includes checkpoint migration and verification:

- Rename `backbone`, `clip_head`, or `frame_head`.
- Replace the canonical sequential head layout.
- Change clip output from 2 logits to 1 logit.
- Change `frame_logits` from 16 values to a tubelet-token length.
- Change label order.
- Save over `best_model.pt`.

## Required Verification For Checkpoint Work

Any task touching model structure or checkpoint loading should verify:

```powershell
python -c "import torch; ckpt=torch.load('PATH_TO_best_model.pt', map_location='cpu'); sd=ckpt['model_state_dict']; print(len([k for k in sd if k.startswith('backbone.')]))"
```

Expected backbone-key count:

```text
182
```

Then run a model load smoke check and a dummy forward pass with:

```text
(B, 16, 3, 224, 224)
```

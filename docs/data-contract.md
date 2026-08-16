# Data Contract

This document defines the canonical data interface for harness work.

## Labels

Clip labels are binary:

- `0 = legal`
- `1 = illegal`

Do not reverse this mapping. Any metric, loss, visualization, or threshold that refers to the illegal class must use class index `1`.

## Source Video Window vs Model Input

The project uses two separate frame concepts:

- Source video window: a segment selected from the original video, commonly 48 frames in the current code and final-package pipeline.
- Model input: 16 sampled frames passed to VideoMAE.

These must not be treated as the same length. A 48-frame source window should be sampled to exactly 16 frames before model inference.

Canonical model input shape:

```text
(B, 16, 3, 224, 224)
```

Single-sample dataset shape:

```text
(16, 3, 224, 224)
```

## Image Preprocessing

Frames must be:

1. Loaded in RGB order when passed to the model.
2. Resized to `224 x 224`.
3. Converted to float in `[0, 1]`.
4. Normalized with ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The final tensor layout is `(T, C, H, W)` per sample and `(B, T, C, H, W)` per batch.

## Frame Sampling

For a source window with `start_frame` and `end_frame`, sample 16 indices evenly across the window. The existing dataset code uses a linspace-style approach and may apply small train-time temporal jitter.

Expected behavior:

- Validation/test sampling should be deterministic.
- Train-time jitter is allowed only if it preserves the 16-frame output length.
- Indices must be clipped to valid video frame bounds.

## Frame Labels

`frame_labels` should align with the 16 sampled model input frames, not the full source window length.

Canonical shape:

```text
(B, 16)
```

For legal clips or clips without matching source events, frame labels are all zeros. For illegal clips, each sampled frame is marked `1.0` when it falls inside a linked event interval.

## Dataset and Batch Interface

The canonical dataset item is a dict:

```python
{
    "pixel_values": Tensor,  # (16, 3, 224, 224)
    "clip_label": Tensor,   # scalar long, 0 legal / 1 illegal
    "frame_labels": Tensor, # (16,)
    "clip_id": str,
    "video_path": str,
}
```

The canonical dataloader batch is also a dict:

```python
{
    "pixel_values": Tensor,  # (B, 16, 3, 224, 224)
    "clip_label": Tensor,   # (B,)
    "frame_labels": Tensor, # (B, 16)
    "clip_id": list[str],
}
```

Tuple batches remain a legacy training detail and should not be used for new harness code.

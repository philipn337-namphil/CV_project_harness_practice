import numpy as np
import torch
import cv2


IMAGE_SIZE = 224

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def canonical_preprocess(frames_rgb):
    resized = np.stack([
        cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        for frame in frames_rgb
    ])

    normalized = (
        resized.astype(np.float32) / 255.0 - IMAGENET_MEAN
    ) / IMAGENET_STD

    return (
        torch.from_numpy(normalized)
        .permute(0, 3, 1, 2)
        .float()
    )


def test_preprocessing_shape_contract():
    frames = np.zeros(
        (16, 360, 640, 3),
        dtype=np.uint8,
    )

    output = canonical_preprocess(frames)

    assert isinstance(output, torch.Tensor)
    assert output.dtype == torch.float32
    assert output.shape == (16, 3, 224, 224)


def test_imagenet_normalization_contract():
    frames = np.zeros(
        (16, 224, 224, 3),
        dtype=np.uint8,
    )

    output = canonical_preprocess(frames)

    expected = torch.tensor(
        -IMAGENET_MEAN / IMAGENET_STD,
        dtype=torch.float32,
    )

    actual = output[0, :, 0, 0]

    assert torch.allclose(
        actual,
        expected,
        atol=1e-5,
    )


def test_white_pixel_normalization_contract():
    frames = np.full(
        (16, 224, 224, 3),
        255,
        dtype=np.uint8,
    )

    output = canonical_preprocess(frames)

    expected = torch.tensor(
        (1.0 - IMAGENET_MEAN) / IMAGENET_STD,
        dtype=torch.float32,
    )

    actual = output[0, :, 0, 0]

    assert torch.allclose(
        actual,
        expected,
        atol=1e-5,
    )
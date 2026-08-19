"""Canonical data pipeline for KHUDA CV."""

from .collate import collate_fn
from .dataset import GarbageDumpingClipDataset

__all__ = [
    "GarbageDumpingClipDataset",
    "collate_fn",
]
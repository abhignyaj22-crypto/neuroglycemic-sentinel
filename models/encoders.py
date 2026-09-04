"""Encoder names — wrappers over lab representations. No GPU auto-grab."""
from __future__ import annotations

SPECIALIST_REPS = ("raw", "pca")
LEARNED_REPS = ("neural", "vq")
FUSION_REPS = ("fused",)


def available_representations(*, include_learned: bool = True, include_fusion: bool = True):
    names = list(SPECIALIST_REPS)
    if include_learned:
        names.extend(LEARNED_REPS)
    if include_fusion:
        names.extend(FUSION_REPS)
    return names

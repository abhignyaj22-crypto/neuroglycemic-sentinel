"""Fusion comparators — not product winners.

Wraps the lab's five strategies plus DNH. Learned CACMF (cross_modal / rep:fused)
is reported as an opponent. Specialists remain primary.
"""
from __future__ import annotations

STRATEGIES = (
    "early",
    "intermediate",
    "late_weighted",
    "attention",
    "cross_modal",
    "dnh_gated",
)

# CACMF training is CPU-locked in this extract.
CACMF_DEVICE = "cpu"


def describe() -> dict:
    return {
        "strategies": list(STRATEGIES),
        "headline": "comparators",
        "cacmf_device": CACMF_DEVICE,
        "note": (
            "Learned fusion loses to the best specialist on the paper tasks. "
            "DNH is a late gate, not a safe held-out guarantee at N≤60."
        ),
    }

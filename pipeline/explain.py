"""Grounded explanation: restates numbers already in the body. Never invents a score."""
from __future__ import annotations

from typing import Any, Dict


def explain(body: Dict[str, Any]) -> Dict[str, Any]:
    status = body.get("status", "ok")
    if status in ("abstained", "blocked") or body.get("probability") is None:
        text = (
            f"The model abstained on {body.get('endpoint', 'this endpoint')}: "
            f"{body.get('abstain_reason') or 'required inputs or labels were not present'}".rstrip(".")
            + "."
        )
        return {"source": "deterministic", "grounded": True, "text": text}
    text = (
        f"{body.get('endpoint')}: score={body.get('probability')} "
        f"({body.get('evidence_status', 'experimental')}). "
        "Research-grade screening, not a diagnosis."
    )
    return {"source": "deterministic", "grounded": True, "text": text}

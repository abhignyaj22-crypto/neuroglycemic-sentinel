"""Fail-closed single-subject infer. Unsynchronized EEG+CGM abstains."""
from __future__ import annotations

from typing import Any, Dict

from data_loaders.pairing import CrossCohortPairingForbidden, refuse_unmatched_eeg_glucose
from pipeline.explain import explain
from pipeline.outcomes import serve_outcome


def _packet(**kwargs) -> Dict[str, Any]:
    out = dict(kwargs)
    out["explanation"] = explain(out)
    return out


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    endpoint = str(payload.get("endpoint") or "stress_detection")
    eeg = str(payload.get("eeg_source") or "")
    wear = str(payload.get("wearable_source") or "")
    try:
        refuse_unmatched_eeg_glucose(eeg, wear)
    except CrossCohortPairingForbidden as exc:
        return _packet(
            endpoint=endpoint,
            status="abstained",
            probability=None,
            abstain_reason=str(exc),
            evidence_status="abstained",
        )

    if endpoint in ("diabetes_complication", "diabetes_progression"):
        pkt = serve_outcome(endpoint, value=None)
        return _packet(
            endpoint=pkt.endpoint,
            status="blocked",
            probability=pkt.value,
            abstain_reason=pkt.abstain_reason,
            evidence_status=pkt.evidence_status.value,
        )

    pkt = serve_outcome(endpoint, value=None)
    return _packet(
        endpoint=pkt.endpoint,
        status="abstained",
        probability=pkt.value,
        abstain_reason=pkt.abstain_reason or "no committed screener in this extract demo",
        evidence_status=pkt.evidence_status.value,
    )

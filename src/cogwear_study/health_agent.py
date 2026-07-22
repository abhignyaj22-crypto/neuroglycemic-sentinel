from typing import Any

import pandas as pd

from .fusion import LateFusion
from .model import LogisticHead


def build_explanation_payload(
    row: pd.Series,
    eeg_head: LogisticHead,
    wearable_head: LogisticHead,
    fusion: LateFusion,
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    """Ground an explanation in model values; do not ask an LLM to invent a score."""
    eeg_contributions = eeg_head.feature_contributions(row).head(top_k)
    wearable_contributions = wearable_head.feature_contributions(row).head(top_k)
    return {
        "patient_id": str(row["patient_id"]),
        "condition": str(row["condition"]),
        "prediction_target": "cognitive_load_during_this_recorded_session",
        "alpha_eeg": float(row["alpha_eeg"]),
        "beta_wearable": float(row["beta_wearable"]),
        "combined_probability": float(row["combined_probability"]),
        "fusion_weights": {
            name: float(weight)
            for name, weight in zip(fusion.modality_names, fusion.weights, strict=True)
        },
        "eeg_linear_contributions": {
            name: float(value) for name, value in eeg_contributions.items()
        },
        "wearable_linear_contributions": {
            name: float(value) for name, value in wearable_contributions.items()
        },
        "limitations": [
            "Research output, not a diagnosis or clinical recommendation.",
            "CogWear labels rest versus Stroop cognitive load; they are not stress, anxiety, depression, or diabetes labels.",
            "The paired pilot analysis has only 10 complete participants, so uncertainty is high.",
        ],
    }


def deterministic_health_agent(payload: dict[str, Any]) -> str:
    weights = payload["fusion_weights"]
    return (
        f"Participant {payload['patient_id']} / {payload['condition']}: "
        f"EEG alpha={payload['alpha_eeg']:.3f}, wearable beta={payload['beta_wearable']:.3f}, "
        f"combined cognitive-load probability={payload['combined_probability']:.3f}. "
        f"Learned fusion weights were EEG={weights['eeg']:.3f} and wearable={weights['wearable']:.3f}. "
        "Interpret only as a small-cohort research result, not an ultimate diagnosis."
    )


def explain_with_langchain(llm: Any, payload: dict[str, Any]) -> str:
    """Optional wording layer. The numeric prediction is fixed before this call."""
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install the optional `llm` dependencies to use this function.") from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Explain only the supplied research-model facts. Do not change numbers, diagnose, "
                "recommend treatment, or infer unobserved conditions.",
            ),
            ("human", "Structured model evidence:\n{payload}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return str(chain.invoke({"payload": payload}))

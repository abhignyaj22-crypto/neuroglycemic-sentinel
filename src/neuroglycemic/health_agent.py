"""HealthAgent with an (optional) LangChain LLM wording step.

Numerical inference always happens before this boundary. The LLM receives a
small, de-identified evidence packet and can only produce qualitative wording;
it cannot calculate, replace, or feed values back into the prediction model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import time
from typing import Any
from uuid import uuid4

from .contracts import HealthEvidencePacket


PROMPT_VERSION = "healthagent-grounded-v1"
_UNSAFE_PHRASES = (
    "you have diabetes",
    "diagnosed with",
    "change your insulin",
    "take insulin",
    "stop taking",
    "medical advice",
)

_DIRECT_IDENTIFIER_KEYS = frozenset(
    {
        "patient_id",
        "patient_identifier",
        "patient_reference",
        "subject_id",
        "hadm_id",
        "stay_id",
        "encounter_id",
        "admission_id",
        "medical_record_number",
        "mrn",
        "person_id",
    }
)

@dataclass(frozen=True)
class HealthAgentTelemetry:
    trace_id: str
    prompt_version: str
    llm_enabled: bool
    llm_called: bool
    provider: str | None
    model: str | None
    latency_ms: float | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    validation_errors: tuple[str, ...]
    fallback_reason: str | None
    raw_response: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HealthAgentResult:
    deterministic_summary: str
    llm_interpretation: str | None
    llm_limitations: tuple[str, ...]
    research_next_step: str | None
    evidence_packet: dict[str, Any]
    telemetry: HealthAgentTelemetry

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HealthAgent:
    """Validate evidence, apply safeguards, and optionally call an injected LLM."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.llm = llm
        self.provider = provider
        self.model_name = model_name

    def run(
        self,
        packet: HealthEvidencePacket,
        *,
        include_raw_response: bool = False,
    ) -> HealthAgentResult:
        trace_id = str(uuid4())
        grounded = packet.as_dict()
        summary = self._deterministic_summary(grounded)
        if self.llm is None:
            telemetry = HealthAgentTelemetry(
                trace_id=trace_id,
                prompt_version=PROMPT_VERSION,
                llm_enabled=False,
                llm_called=False,
                provider=self.provider,
                model=self.model_name,
                latency_ms=None,
                request_id=None,
                input_tokens=None,
                output_tokens=None,
                validation_errors=(),
                fallback_reason="LLM not configured; deterministic renderer used.",
                raw_response=None,
            )
            return HealthAgentResult(summary, None, (), None, grounded, telemetry)

        sanitized = self._sanitize(grounded)
        started = time.perf_counter()
        raw = ""
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
        request_id: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        try:
            raw, response_metadata = self._invoke_langchain(sanitized)
            request_id = response_metadata.get("request_id")
            input_tokens = response_metadata.get("input_tokens")
            output_tokens = response_metadata.get("output_tokens")
            parsed = json.loads(raw)
            errors.extend(self._validate_llm_payload(parsed, sanitized))
        except Exception as exc:  # a failed wording layer must not fail inference
            errors.append(f"{type(exc).__name__}: {exc}")
        latency_ms = (time.perf_counter() - started) * 1000.0

        fallback_reason = None
        interpretation: str | None = None
        limitations: tuple[str, ...] = ()
        next_step: str | None = None
        if errors or parsed is None:
            fallback_reason = "LLM response failed grounding/schema validation."
        else:
            interpretation = str(parsed["interpretation"])
            limitations = tuple(str(value) for value in parsed["limitations"])
            next_step = str(parsed["research_next_step"])
        telemetry = HealthAgentTelemetry(
            trace_id=trace_id,
            prompt_version=PROMPT_VERSION,
            llm_enabled=True,
            llm_called=True,
            provider=self.provider or type(self.llm).__module__.split(".")[0],
            model=self.model_name or getattr(self.llm, "model_name", None),
            latency_ms=latency_ms,
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            validation_errors=tuple(errors),
            fallback_reason=fallback_reason,
            raw_response=raw if include_raw_response else None,
        )
        return HealthAgentResult(
            deterministic_summary=summary,
            llm_interpretation=interpretation,
            llm_limitations=limitations,
            research_next_step=next_step,
            evidence_packet=grounded,
            telemetry=telemetry,
        )

    @staticmethod
    def _deterministic_summary(packet: dict[str, Any]) -> str:
        output = packet["model_output"]
        rendered: list[str] = []
        for name, value in output.items():
            if isinstance(value, float):
                rendered.append(f"{name}={value:.3f}")
            elif isinstance(value, (str, int, bool)) or value is None:
                rendered.append(f"{name}={value}")
        return (
            f"Research task {packet['task']} at {packet['anchor_time']}: "
            + ", ".join(rendered)
            + f". Release status: {packet['release_status']}."
        )

    @staticmethod
    def _sanitize(packet: dict[str, Any]) -> dict[str, Any]:
        safe = json.loads(json.dumps(packet))
        patient_id = str(safe.pop("patient_id"))

        safe["metadata"] = {
            key: value
            for key, value in safe.get("metadata", {}).items()
            if key
            in {
                "cohort",
                "horizon_hours",
                "model_version",
                "data_scope",
            }
        }

        safe = _recursively_deidentify(
            safe,
            patient_id=patient_id,
        )
        safe["patient_reference"] = hashlib.sha256(
            patient_id.encode("utf-8")
        ).hexdigest()[:12]
        return safe

    def _invoke_langchain(
        self,
        packet: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Install requirements-llm.txt to enable the HealthAgent LLM call."
            ) from exc
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the wording component of a research-only HealthAgent. Use only the "
                    "supplied JSON facts. Do not calculate or change scores, infer missing conditions, "
                    "diagnose, recommend treatment, or add numerical claims. Return JSON only with "
                    "exactly these keys: interpretation (string), limitations (array of strings), "
                    "research_next_step (string).",
                ),
                ("human", "Frozen model evidence:\n{packet}"),
            ]
        )
        formatted = prompt.invoke({"packet": json.dumps(packet, sort_keys=True)})
        response = self.llm.invoke(formatted)
        result = StrOutputParser().invoke(response)
        metadata: dict[str, Any] = {}
        source_metadata = getattr(response, "response_metadata", {}) or {}
        usage_metadata = getattr(response, "usage_metadata", {}) or {}
        usage = (
            usage_metadata
            or source_metadata.get("token_usage", {})
            or source_metadata.get("usage", {})
            or {}
        )
        metadata["request_id"] = source_metadata.get("request_id") or source_metadata.get("id")
        metadata["input_tokens"] = usage.get("prompt_tokens") or usage.get("input_tokens")
        metadata["output_tokens"] = usage.get("completion_tokens") or usage.get("output_tokens")
        return str(result), metadata

    @staticmethod
    def _validate_llm_payload(
        payload: Any, grounded_packet: dict[str, Any]
    ) -> list[str]:
        errors: list[str] = []
        required = {"interpretation", "limitations", "research_next_step"}
        if not isinstance(payload, dict):
            return ["Response must be a JSON object."]
        if set(payload) != required:
            errors.append(f"Response keys must be exactly {sorted(required)}.")
        if not isinstance(payload.get("interpretation"), str):
            errors.append("interpretation must be a string.")
        if not isinstance(payload.get("limitations"), list) or not all(
            isinstance(value, str) for value in payload.get("limitations", [])
        ):
            errors.append("limitations must be an array of strings.")
        if not isinstance(payload.get("research_next_step"), str):
            errors.append("research_next_step must be a string.")
        text = json.dumps(payload).lower()
        for phrase in _UNSAFE_PHRASES:
            if phrase in text:
                errors.append(f"Unsafe phrase in response: {phrase!r}.")
        allowed_numbers = _collect_numbers(json.dumps(grounded_packet))
        for number in _collect_numbers(json.dumps(payload)):
            if not any(math.isclose(number, allowed, rel_tol=1e-6, abs_tol=1e-6) for allowed in allowed_numbers):
                errors.append(f"Ungrounded numerical claim: {number:g}.")
        return errors


def _collect_numbers(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text))

def _recursively_deidentify(
    value: Any,
    *,
    patient_id: str,
) -> Any:
    """Remove identifiers from nested JSON-compatible evidence."""

    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}

        for key, item in value.items():
            normalized_key = str(key).strip().lower()

            if normalized_key in _DIRECT_IDENTIFIER_KEYS:
                continue

            cleaned[str(key)] = _recursively_deidentify(
                item,
                patient_id=patient_id,
            )

        return cleaned

    if isinstance(value, list):
        return [
            _recursively_deidentify(item, patient_id=patient_id)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _recursively_deidentify(item, patient_id=patient_id)
            for item in value
        )

    if isinstance(value, str) and patient_id:
        return value.replace(
            patient_id,
            "[redacted-patient]",
        )

    return value

def build_openai_llm(*, model: str, temperature: float = 0.0) -> Any:
    """Construct the optional LangChain OpenAI chat client at the CLI boundary."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Install requirements-llm.txt to use --use-llm.") from exc
    return ChatOpenAI(model=model, temperature=temperature)

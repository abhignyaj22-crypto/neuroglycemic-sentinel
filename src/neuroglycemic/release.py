"""Checkpoint release manifests for safe neural serving.

The checkpoint stores trainable state.  This sidecar stores the independent
evaluation decision.  Keeping them separate makes it impossible to label a
random or merely loadable checkpoint as an evaluated model by changing its
in-memory service configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


RELEASE_MANIFEST_SCHEMA = "neuroglycemic-model-release-v1"
RELEASE_STATUSES = {"approved", "research_only", "rejected"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_manifest_path(checkpoint_path: Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_suffix(path.suffix + ".release.json")


@dataclass(frozen=True)
class ModelReleaseManifest:
    schema_version: str
    checkpoint_sha256: str
    status: str
    patient_disjoint_evaluation: bool
    cohorts: tuple[str, ...]
    decision_reasons: tuple[str, ...]
    metrics_file: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_MANIFEST_SCHEMA:
            raise ValueError("Unsupported model release manifest schema.")
        if self.status not in RELEASE_STATUSES:
            raise ValueError(f"Unsupported model release status {self.status!r}.")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest.")
        try:
            int(self.checkpoint_sha256, 16)
        except ValueError as exc:
            raise ValueError("checkpoint_sha256 must be hexadecimal.") from exc
        if not self.cohorts or any(not value.strip() for value in self.cohorts):
            raise ValueError("At least one evaluated cohort is required.")
        if not self.decision_reasons or any(
            not value.strip() for value in self.decision_reasons
        ):
            raise ValueError("At least one release decision reason is required.")
        if self.status == "approved" and not self.patient_disjoint_evaluation:
            raise ValueError("Approved models require patient-disjoint evaluation.")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["cohorts"] = list(self.cohorts)
        values["decision_reasons"] = list(self.decision_reasons)
        return values

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelReleaseManifest":
        return cls(
            schema_version=str(values["schema_version"]),
            checkpoint_sha256=str(values["checkpoint_sha256"]),
            status=str(values["status"]),
            patient_disjoint_evaluation=bool(values["patient_disjoint_evaluation"]),
            cohorts=tuple(str(value) for value in values["cohorts"]),
            decision_reasons=tuple(str(value) for value in values["decision_reasons"]),
            metrics_file=(
                None if values.get("metrics_file") is None else str(values["metrics_file"])
            ),
        )


def write_release_manifest(
    checkpoint_path: Path,
    *,
    status: str,
    patient_disjoint_evaluation: bool,
    cohorts: tuple[str, ...] | list[str],
    decision_reasons: tuple[str, ...] | list[str],
    metrics_file: str | None = None,
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = ModelReleaseManifest(
        schema_version=RELEASE_MANIFEST_SCHEMA,
        checkpoint_sha256=file_sha256(checkpoint),
        status=status,
        patient_disjoint_evaluation=patient_disjoint_evaluation,
        cohorts=tuple(cohorts),
        decision_reasons=tuple(decision_reasons),
        metrics_file=metrics_file,
    )
    destination = release_manifest_path(checkpoint)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_release_manifest(
    checkpoint_path: Path, *, manifest_path: Path | None = None
) -> ModelReleaseManifest:
    checkpoint = Path(checkpoint_path).resolve()
    destination = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else release_manifest_path(checkpoint)
    )
    if not destination.is_file():
        raise ValueError(
            "Checkpoint has no evaluation release manifest. Run held-out evaluation "
            "or explicitly use the research-only service path."
        )
    raw = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model release manifest must contain a JSON object.")
    manifest = ModelReleaseManifest.from_mapping(raw)
    if manifest.checkpoint_sha256 != file_sha256(checkpoint):
        raise ValueError("Release manifest checkpoint SHA-256 does not match the model.")
    return manifest

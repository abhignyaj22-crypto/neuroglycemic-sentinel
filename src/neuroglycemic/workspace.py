"""External runtime workspace for protected data and derived artifacts.

The Git repository contains software only.  Raw data, aligned windows, model
checkpoints, run manifests, and figures live under a caller-selected directory
that must not be the repository or one of its descendants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ResearchWorkspace:
    root: Path
    raw: Path
    canonical: Path
    aligned: Path
    models: Path
    runs: Path
    figures: Path

    @classmethod
    def create(cls, root: Path, *, repository_root: Path) -> "ResearchWorkspace":
        resolved_root = Path(root).expanduser().resolve()
        resolved_repository = Path(repository_root).expanduser().resolve()
        if (
            resolved_root == resolved_repository
            or _is_relative_to(resolved_root, resolved_repository)
            or _is_relative_to(resolved_repository, resolved_root)
        ):
            raise ValueError(
                "The research workspace must be outside and a disjoint sibling of "
                "the software repository, not its parent or descendant. "
                f"resolved_workspace={resolved_root}; "
                f"resolved_repository={resolved_repository}. "
                "Choose a path such as ../neuroglycemic-runtime."
            )
        paths = {
            name: resolved_root / name
            for name in ("raw", "canonical", "aligned", "models", "runs", "figures")
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return cls(root=resolved_root, **paths)

    def run_directory(self, run_name: str) -> Path:
        clean = run_name.strip()
        if not clean or clean in {".", ".."} or "/" in clean or "\\" in clean:
            raise ValueError("run_name must be one non-empty path component.")
        destination = self.runs / clean
        destination.mkdir(parents=True, exist_ok=True)
        return destination

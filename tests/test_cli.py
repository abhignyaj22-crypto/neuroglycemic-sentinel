from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as application  # noqa: E402


def test_cli_requires_an_explicit_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"])

    with pytest.raises(SystemExit) as error:
        application.cli()

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "the following arguments are required: study" in captured.err
    assert "train-neural" in captured.err


def test_cli_help_marks_legacy_research_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])

    with pytest.raises(SystemExit) as error:
        application.cli()

    captured = capsys.readouterr()
    assert error.value.code == 0
    assert "No implicit study is run" in captured.out
    assert "legacy research commands" in captured.out


def test_missing_neural_data_suggests_close_runtime_filename(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    aligned = tmp_path / "runtime" / "aligned"
    aligned.mkdir(parents=True)
    correct = aligned / "mimiciv_demo_neural_windows.csv.gz"
    correct.write_bytes(b"fixture")
    typo = aligned / "mimiv_demo_neural_windows.csv.gz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "evaluate-neural",
            "--data",
            str(typo),
            "--workspace",
            str(tmp_path / "runtime"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        application.cli()

    assert error.value.code == 2
    assert f"Did you mean: {correct}" in capsys.readouterr().err


def test_prepare_mimic_rebuild_is_explicit_and_atomic(
    tmp_path: Path, monkeypatch
) -> None:
    from src.neuroglycemic import mimic_neural

    source = tmp_path / "source.csv"
    source.write_text("fixture", encoding="utf-8")
    workspace = tmp_path / "runtime"
    frame = pd.DataFrame({"patient_id": ["p1"], "value": [1.0]})
    monkeypatch.setattr(mimic_neural, "prepare_mimic_neural_file", lambda _: frame)
    arguments = [
        "main.py",
        "prepare-mimic-neural",
        "--data",
        str(source),
        "--workspace",
        str(workspace),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    application.cli()
    destination = workspace / "aligned" / "mimiciv_demo_neural_windows.csv.gz"
    assert destination.is_file()

    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit):
        application.cli()

    replacement = pd.DataFrame({"patient_id": ["p2"], "value": [2.0]})
    monkeypatch.setattr(
        mimic_neural, "prepare_mimic_neural_file", lambda _: replacement
    )
    monkeypatch.setattr(sys, "argv", [*arguments, "--rebuild"])
    application.cli()
    written = pd.read_csv(destination)
    assert written["patient_id"].tolist() == ["p2"]
    assert not destination.with_suffix(destination.suffix + ".tmp").exists()

from __future__ import annotations

import sys
from pathlib import Path

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


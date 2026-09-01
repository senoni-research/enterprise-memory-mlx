from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx import cli
from enterprise_memory_mlx.legacy_guard import (
    LEGACY_COMMANDS,
    LegacyPipelineDisabledError,
    block_legacy_command,
)


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["compile"], "_compile"),
        (["train", "--dry-run"], "_train"),
        (["evaluate"], "_evaluate"),
        (["route", "test query"], "_route"),
        (["chat"], "_chat"),
    ],
)
def test_cli_blocks_legacy_before_handler(
    argv: list[str],
    handler: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def must_not_run(*_args, **_kwargs) -> None:
        raise AssertionError(f"{handler} must not run")

    monkeypatch.setattr(cli, handler, must_not_run)

    assert cli.main(argv) == 2
    output = capsys.readouterr().out
    assert "disabled" in output
    assert "scientifically invalid" in output
    assert "There is no override flag" in output


def test_every_legacy_command_is_covered() -> None:
    assert {"compile", "train", "evaluate", "route", "chat"} == LEGACY_COMMANDS


def test_guard_rejects_unknown_command() -> None:
    with pytest.raises(ValueError, match="Unknown legacy command"):
        block_legacy_command("benchmark")


def test_guard_raises_dedicated_error() -> None:
    with pytest.raises(LegacyPipelineDisabledError, match="leaky splits"):
        block_legacy_command("train")


def test_help_marks_legacy_commands_disabled() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert help_text.count("DISABLED legacy") == 5


def test_doctor_reports_block_without_recommending_legacy_preset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "doctor_report",
        lambda: {
            "hardware": {
                "system": "Darwin",
                "machine": "arm64",
                "chip": "Apple M4 Max",
                "memory_gib": 128.0,
                "gpu_cores": 40,
                "os_version": "26.0",
                "is_apple_silicon": True,
            },
            "mlx_lm_installed": False,
            "mlx_lm_version": None,
            "python": "3.13",
            "recommended_preset": {
                "name": "must-not-be-shown",
                "model": "must-not-be-shown",
            },
        },
    )

    assert cli.main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "Training status" in output
    assert "BLOCKED" in output
    assert "must-not-be-shown" not in output


def test_legacy_shell_scripts_are_hard_stops(project_root: Path) -> None:
    for relative in ("scripts/run_demo.sh", "scripts/run_ablation.sh"):
        text = (project_root / relative).read_text(encoding="utf-8")
        assert "scientifically invalid legacy pipeline" in text
        assert "exit 2" in text
        assert "emmlx train" not in text


def test_makefile_legacy_targets_are_hard_stops(project_root: Path) -> None:
    text = (project_root / "Makefile").read_text(encoding="utf-8")
    for target in ("compile", "dry-run", "train", "ablation", "evaluate", "route", "chat"):
        block = text.split(f"\n{target}:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
        assert "$(LEGACY_DISABLED)" in block
        assert "exit 2" in block

from __future__ import annotations

from pathlib import Path

import pytest

from enterprise_memory_mlx.cli import build_parser
from enterprise_memory_mlx.grpo_protocol import write_frozen_protocol
from enterprise_memory_mlx.grpo_runtime import run_execute, validate_experiment


def test_grpo_pilot_cli_actions_exist() -> None:
    parser = build_parser()
    args = parser.parse_args(["grpo-pilot", "validate"])
    assert args.command == "grpo-pilot"
    assert args.grpo_action == "validate"
    for action in ("preflight", "execute", "evaluate", "prepare-review"):
        parsed = parser.parse_args(["grpo-pilot", action])
        assert parsed.grpo_action == action


def test_execute_refuses_without_preflight(isolated_project: Path) -> None:
    write_frozen_protocol(isolated_project)
    validate_experiment(isolated_project)
    with pytest.raises(RuntimeError, match="preflight has not passed"):
        run_execute(isolated_project)

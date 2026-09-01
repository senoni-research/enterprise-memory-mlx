"""Fail-closed guard for the scientifically invalid legacy workflow."""

from __future__ import annotations

LEGACY_COMMANDS: frozenset[str] = frozenset(
    {"compile", "train", "evaluate", "route", "chat"}
)

LEGACY_DISABLED_MESSAGE = (
    "The '{command}' command is disabled: it belongs to the legacy pipeline, "
    "which is scientifically invalid because it uses leaky splits, q/v-only "
    "partial-layer LoRA, fixed iteration counts, mis-calibrated MLX LoRA scale, "
    "and lexical answer scoring. Implement and validate the revised acquisition "
    "contract before enabling this capability. There is no override flag."
)


LEGACY_LIBRARY_MESSAGE = (
    "{component} belongs to the scientifically invalid legacy pipeline and is "
    "disabled for experiments and deployment. Historical-reference tests may "
    "pass allow_scientifically_invalid=True to document the old behaviour; "
    "nothing else may."
)


class LegacyPipelineDisabledError(RuntimeError):
    """Raised before any legacy pipeline code can execute."""


def block_legacy_command(command: str) -> None:
    if command not in LEGACY_COMMANDS:
        raise ValueError(f"Unknown legacy command: {command}")
    raise LegacyPipelineDisabledError(LEGACY_DISABLED_MESSAGE.format(command=command))


def guard_legacy_component(component: str, *, allow_scientifically_invalid: bool) -> None:
    """Fail closed unless a caller explicitly acknowledges scientific invalidity."""
    if not allow_scientifically_invalid:
        raise LegacyPipelineDisabledError(
            LEGACY_LIBRARY_MESSAGE.format(component=component)
        )

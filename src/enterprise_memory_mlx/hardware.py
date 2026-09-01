from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    system: str
    machine: str
    chip: str
    memory_gib: float
    gpu_cores: int | None
    os_version: str

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine == "arm64"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["is_apple_silicon"] = self.is_apple_silicon
        return value


@dataclass(frozen=True)
class TrainingPreset:
    name: str
    model: str
    num_layers: int
    batch_size: int
    grad_accumulation_steps: int
    max_seq_length: int
    grad_checkpoint: bool
    inject_iters: int
    align_iters: int
    recover_iters: int
    inject_lr: float
    align_lr: float
    recover_lr: float
    lora_rank: int = 8
    lora_scale: float = 20.0


PRESETS: dict[str, TrainingPreset] = {
    "smoke": TrainingPreset(
        name="smoke",
        model="mlx-community/Llama-3.2-1B-Instruct-4bit",
        num_layers=4,
        batch_size=1,
        grad_accumulation_steps=1,
        max_seq_length=512,
        grad_checkpoint=True,
        inject_iters=20,
        align_iters=30,
        recover_iters=10,
        inject_lr=2e-5,
        align_lr=2e-5,
        recover_lr=5e-6,
    ),
    "m4max-quick": TrainingPreset(
        name="m4max-quick",
        model="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        num_layers=8,
        batch_size=1,
        grad_accumulation_steps=2,
        max_seq_length=768,
        grad_checkpoint=True,
        inject_iters=100,
        align_iters=140,
        recover_iters=50,
        inject_lr=2e-5,
        align_lr=2e-5,
        recover_lr=4e-6,
    ),
    "m4max-balanced": TrainingPreset(
        name="m4max-balanced",
        model="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        num_layers=16,
        batch_size=1,
        grad_accumulation_steps=4,
        max_seq_length=1024,
        grad_checkpoint=True,
        inject_iters=250,
        align_iters=320,
        recover_iters=100,
        inject_lr=2e-5,
        align_lr=1.5e-5,
        recover_lr=3e-6,
        lora_rank=16,
        lora_scale=32.0,
    ),
    "m4max-large": TrainingPreset(
        name="m4max-large",
        model="mlx-community/Qwen3-4B-Instruct-2507-bf16",
        num_layers=24,
        batch_size=1,
        grad_accumulation_steps=4,
        max_seq_length=1536,
        grad_checkpoint=True,
        inject_iters=300,
        align_iters=400,
        recover_iters=140,
        inject_lr=1.5e-5,
        align_lr=1.2e-5,
        recover_lr=2e-6,
        lora_rank=16,
        lora_scale=32.0,
    ),
}


def detect_hardware() -> HardwareProfile:
    system = platform.system()
    machine = platform.machine()
    chip = platform.processor() or "unknown"
    memory_bytes = 0
    gpu_cores: int | None = None

    if system == "Darwin":
        memory_bytes = _sysctl_int("hw.memsize")
        chip = _mac_chip_name() or chip
        gpu_cores = _mac_gpu_cores()
    else:
        try:
            page_size = int(subprocess.check_output(["getconf", "PAGE_SIZE"], text=True).strip())
            pages = int(subprocess.check_output(["getconf", "_PHYS_PAGES"], text=True).strip())
            memory_bytes = page_size * pages
        except (OSError, ValueError, subprocess.SubprocessError):
            memory_bytes = 0

    memory_gib = round(memory_bytes / (1024**3), 1) if memory_bytes else 0.0
    return HardwareProfile(
        system=system,
        machine=machine,
        chip=chip,
        memory_gib=memory_gib,
        gpu_cores=gpu_cores,
        os_version=platform.mac_ver()[0] if system == "Darwin" else platform.release(),
    )


def resolve_preset(name: str, hardware: HardwareProfile | None = None) -> TrainingPreset:
    if name != "auto":
        try:
            return PRESETS[name]
        except KeyError as exc:
            choices = ", ".join(["auto", *PRESETS])
            raise ValueError(f"Unknown preset {name!r}. Choose one of: {choices}") from exc

    profile = hardware or detect_hardware()
    if profile.memory_gib and profile.memory_gib < 48:
        return PRESETS["m4max-quick"]
    if profile.memory_gib and profile.memory_gib >= 96:
        return PRESETS["m4max-large"]
    return PRESETS["m4max-balanced"]


def _sysctl_int(name: str) -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", name], text=True).strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _mac_chip_name() -> str | None:
    try:
        raw = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"], text=True
        )
        data = json.loads(raw)
        rows = data.get("SPHardwareDataType", [])
        if rows:
            return rows[0].get("chip_type") or rows[0].get("cpu_type")
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        return None
    return None


def _mac_gpu_cores() -> int | None:
    try:
        output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Total Number of Cores:\s*(\d+)", output)
    return int(match.group(1)) if match else None

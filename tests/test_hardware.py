from enterprise_memory_mlx.hardware import HardwareProfile, resolve_preset


def profile(memory: float) -> HardwareProfile:
    return HardwareProfile(
        system="Darwin",
        machine="arm64",
        chip="Apple M4 Max",
        memory_gib=memory,
        gpu_cores=40,
        os_version="15.6",
    )


def test_auto_preset_for_m4_max_memory_tiers() -> None:
    assert resolve_preset("auto", profile(36)).name == "m4max-quick"
    assert resolve_preset("auto", profile(64)).name == "m4max-balanced"
    assert resolve_preset("auto", profile(128)).name == "m4max-large"

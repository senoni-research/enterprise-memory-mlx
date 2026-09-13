from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from enterprise_memory_mlx.benchmark import (
    MAX_GENERATION_OUTPUT_TOKENS,
    BenchmarkConfig,
    MLXBenchmarkBackend,
)
from enterprise_memory_mlx.experiment_profiles import MODEL_UPGRADE_CONFIG


class _FakeTokenizer:
    chat_template = "fake-template"

    def __init__(self, assistant_prefix: str) -> None:
        self.assistant_prefix = assistant_prefix

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert isinstance(enable_thinking, bool)
        conversation = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return conversation + (self.assistant_prefix if add_generation_prompt else "")

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[str]:
        assert add_special_tokens is False
        return text.split()


def _install_fake_mlx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assistant_prefix: str = "<|assistant|>\n",
    generated_text: str = '{"ok":true}',
    finish_reason: str = "stop",
) -> dict[str, object]:
    calls: dict[str, object] = {"stream_count": 0}
    tokenizer = _FakeTokenizer(assistant_prefix)

    def load(model_name: str, *, revision: str, adapter_path=None):
        assert model_name == "fake/model"
        assert revision == "pinned-revision"
        assert adapter_path is None
        return object(), tokenizer

    def stream_generate(model, selected_tokenizer, **kwargs):
        del model
        assert selected_tokenizer is tokenizer
        calls["stream_count"] = int(calls["stream_count"]) + 1
        calls["max_tokens"] = kwargs["max_tokens"]
        yield SimpleNamespace(
            text=generated_text,
            finish_reason=finish_reason,
            prompt_tokens=None,
            generation_tokens=len(generated_text),
            peak_memory=1.25,
        )

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = load
    mlx_lm.stream_generate = stream_generate
    models = ModuleType("mlx_lm.models")
    cache = ModuleType("mlx_lm.models.cache")
    cache.make_prompt_cache = lambda model: {"model": model}
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda *, temp: {"temperature": temp}

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    return calls


def _backend(*, max_context_tokens: int = 10_000) -> MLXBenchmarkBackend:
    return MLXBenchmarkBackend(
        "fake/model",
        revision="pinned-revision",
        max_context_tokens=max_context_tokens,
    )


@pytest.mark.parametrize(
    "source_text",
    [
        'source_code = "<think>"',
        'quoted_balanced = "<think>literal</think>"',
        'quoted_unbalanced = "literal </think>"',
    ],
)
def test_composed_backend_ignores_think_tags_inside_supplied_source(
    monkeypatch: pytest.MonkeyPatch,
    source_text: str,
) -> None:
    calls = _install_fake_mlx(monkeypatch)
    backend = _backend()

    answer = backend.generate(
        system_prompt="Review only the supplied source.",
        question=source_text,
        max_tokens=MAX_GENERATION_OUTPUT_TOKENS,
    )

    assert answer.output == '{"ok":true}'
    assert answer.parse_status == "plain"
    assert calls["stream_count"] == 1
    assert calls["max_tokens"] == 4096


@pytest.mark.parametrize(
    "generated_text",
    [
        '{"quoted":"<think>"}',
        '{"quoted":"<think>literal</think>"}',
        '{"quoted":"literal </think>"}',
    ],
)
def test_composed_backend_treats_tags_inside_valid_json_strings_as_plain_output(
    monkeypatch: pytest.MonkeyPatch,
    generated_text: str,
) -> None:
    _install_fake_mlx(monkeypatch, generated_text=generated_text)
    backend = _backend()

    answer = backend.generate(
        system_prompt="Return JSON.",
        question="Review the source.",
        max_tokens=512,
    )

    assert answer.output == generated_text
    assert answer.parse_status == "plain"


def test_composed_backend_rejects_genuinely_invalid_assistant_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_mlx(
        monkeypatch,
        assistant_prefix="<|assistant|>\n<think>\n",
    )
    backend = _backend()

    with pytest.raises(RuntimeError, match="assistant prefix left an open thinking channel"):
        backend.generate(
            system_prompt="Review.",
            question="Source.",
            max_tokens=512,
        )

    assert calls["stream_count"] == 0


def test_composed_backend_rejects_genuine_unexpected_reasoning_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '<think>private reasoning</think>{"ok":true}'
    _install_fake_mlx(monkeypatch, generated_text=raw)
    backend = _backend()

    answer = backend.generate(
        system_prompt="Return JSON.",
        question="Review.",
        max_tokens=512,
    )

    assert answer.output is None
    assert answer.raw_output == raw
    assert answer.parse_status == "thinking_protocol_violation"


def test_composed_backend_fails_when_prompt_plus_reserve_exceeds_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_mlx(monkeypatch)
    backend = _backend(max_context_tokens=4_096)

    with pytest.raises(ValueError, match="plus reserved output exceeds model context"):
        backend.generate(
            system_prompt="Review.",
            question="Source.",
            max_tokens=4_096,
        )

    assert calls["stream_count"] == 0


def test_new_output_cap_is_configurable_without_changing_historical_profile() -> None:
    assert BenchmarkConfig(max_output_tokens=4_096).max_output_tokens == 4_096
    with pytest.raises(ValueError, match="cannot exceed 4096"):
        BenchmarkConfig(max_output_tokens=4_097)
    assert MODEL_UPGRADE_CONFIG.generator_max_output_tokens == 1_024

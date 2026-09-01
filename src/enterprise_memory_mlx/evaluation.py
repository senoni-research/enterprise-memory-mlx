from __future__ import annotations

import gc
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compiler import SYSTEM_PROMPT
from .utils import atomic_write_text, read_jsonl


@dataclass(frozen=True)
class Score:
    passed: bool
    exact_or_contains: bool
    token_f1: float
    keyword_coverage: float


def evaluate_models(
    *,
    model_name: str,
    suite_path: Path,
    output_dir: Path,
    adapter_path: Path | None,
    include_base: bool = True,
    max_tokens: int = 220,
    allow_scientifically_invalid: bool = False,
) -> Path:
    from .legacy_guard import guard_legacy_component

    guard_legacy_component(
        "evaluation.evaluate_models",
        allow_scientifically_invalid=allow_scientifically_invalid,
    )
    suite = read_jsonl(suite_path)
    if not suite:
        raise ValueError(f"Evaluation suite is empty: {suite_path}")

    results: list[dict[str, Any]] = []
    if include_base:
        results.append(
            _evaluate_one(
                label="base",
                model_name=model_name,
                adapter_path=None,
                suite=suite,
                max_tokens=max_tokens,
            )
        )
        _release_mlx_memory()
    if adapter_path is not None:
        results.append(
            _evaluate_one(
                label="adapter",
                model_name=model_name,
                adapter_path=adapter_path,
                suite=suite,
                max_tokens=max_tokens,
            )
        )
        _release_mlx_memory()

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model": model_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "suite": str(suite_path),
        "results": results,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"evaluation-{timestamp}.json"
    markdown_path = output_dir / f"evaluation-{timestamp}.md"
    atomic_write_text(json_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(markdown_path, _render_markdown(payload))
    return markdown_path


def _evaluate_one(
    *,
    label: str,
    model_name: str,
    adapter_path: Path | None,
    suite: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError('MLX-LM is required. Install with: pip install -e ".[mac]"') from exc

    model, tokenizer = load(
        model_name,
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    sampler = make_sampler(temp=0.0)
    rows: list[dict[str, Any]] = []
    for item in suite:
        question = str(item["question"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        answer = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        score = score_answer(
            output=str(answer),
            expected=str(item.get("expected", "")),
            keywords=[str(value) for value in item.get("keywords", [])],
            kind=str(item.get("kind", "known")),
            allow_scientifically_invalid=True,  # guarded at evaluate_models entry
        )
        rows.append(
            {
                **item,
                "output": str(answer).strip(),
                "score": {
                    "passed": score.passed,
                    "exact_or_contains": score.exact_or_contains,
                    "token_f1": round(score.token_f1, 4),
                    "keyword_coverage": round(score.keyword_coverage, 4),
                },
            }
        )

    passed = sum(1 for row in rows if row["score"]["passed"])
    return {
        "label": label,
        "summary": {
            "passed": passed,
            "total": len(rows),
            "accuracy": round(passed / len(rows), 4),
            "mean_token_f1": round(
                sum(row["score"]["token_f1"] for row in rows) / len(rows), 4
            ),
            "mean_keyword_coverage": round(
                sum(row["score"]["keyword_coverage"] for row in rows) / len(rows), 4
            ),
        },
        "rows": rows,
    }


def score_answer(
    *,
    output: str,
    expected: str,
    keywords: list[str],
    kind: str = "known",
    allow_scientifically_invalid: bool = False,
) -> Score:
    from .legacy_guard import guard_legacy_component

    guard_legacy_component(
        "evaluation.score_answer",
        allow_scientifically_invalid=allow_scientifically_invalid,
    )
    output_normalized = normalize_text(output)
    expected_normalized = normalize_text(expected)
    contains = bool(expected_normalized) and (
        expected_normalized in output_normalized or output_normalized in expected_normalized
    )
    f1 = token_f1(output_normalized, expected_normalized)
    coverage = keyword_coverage(output_normalized, keywords)

    if kind == "unknown":
        refusal_markers = (
            "do not have",
            "dont have",
            "no authoritative",
            "check the current",
            "check the source",
            "cannot verify",
        )
        passed = any(marker in output_normalized for marker in refusal_markers)
    else:
        passed = contains or f1 >= 0.55 or (len(keywords) >= 2 and coverage >= 0.75)
    return Score(
        passed=passed,
        exact_or_contains=contains,
        token_f1=f1,
        keyword_coverage=coverage,
    )


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).lower()
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = re.sub(r"[^a-z0-9£$€%]+", " ", value)
    return " ".join(value.split())


def token_f1(output: str, expected: str) -> float:
    output_tokens = output.split()
    expected_tokens = expected.split()
    if not output_tokens or not expected_tokens:
        return 0.0
    output_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    for token in output_tokens:
        output_counts[token] = output_counts.get(token, 0) + 1
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    overlap = sum(
        min(count, expected_counts.get(token, 0))
        for token, count in output_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(output_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(output: str, keywords: list[str]) -> float:
    normalized_keywords = [normalize_text(item) for item in keywords if normalize_text(item)]
    if not normalized_keywords:
        return 0.0
    hits = sum(1 for keyword in normalized_keywords if keyword in output)
    return hits / len(normalized_keywords)


def _release_mlx_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Enterprise memory evaluation",
        "",
        f"- Model: `{payload['model']}`",
        f"- Adapter: `{payload['adapter_path'] or 'none'}`",
        f"- Suite: `{payload['suite']}`",
        "",
        "| Run | Passed | Total | Accuracy | Mean token F1 | Keyword coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        summary = result["summary"]
        lines.append(
            f"| {result['label']} | {summary['passed']} | {summary['total']} | "
            f"{summary['accuracy']:.1%} | {summary['mean_token_f1']:.3f} | "
            f"{summary['mean_keyword_coverage']:.3f} |"
        )
    lines.extend(["", "## Failures", ""])
    failures = []
    for result in payload["results"]:
        for row in result["rows"]:
            if not row["score"]["passed"]:
                failures.append((result["label"], row))
    if not failures:
        lines.append("No failures under the configured scoring policy.")
    else:
        for label, row in failures:
            lines.extend(
                [
                    f"### {label}: {row.get('record_id') or row.get('kind')}",
                    "",
                    f"**Question:** {row['question']}",
                    "",
                    f"**Expected:** {row['expected']}",
                    "",
                    f"**Output:** {row['output']}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"

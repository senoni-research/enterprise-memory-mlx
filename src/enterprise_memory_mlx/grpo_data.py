"""Compile a leakage-resistant GRPO dataset from frozen v3/v4 cases."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .compiler import load_records
from .specialization_fact_state_experiment import (
    AMENDED_SYSTEM_PROMPT,
    STATE_AUTHORITY_GUIDE,
)
from .specialization_remedy_experiment import (
    CHALLENGER_SYSTEM_PROMPT,
    _source_dict,
)
from .utils import read_jsonl, sha256_json, sha256_text

PROTOCOL_ID = "company-task-specialization/qwen4b-grpo-v1"
V3_CASES_RELATIVE = (
    "knowledge/company_task_specialization/v3_remedy_logic/qualification_cases.jsonl"
)
V4_CASES_RELATIVE = "knowledge/company_task_specialization/v4_fact_state/contrast_cases.jsonl"
V4_POLICY_RELATIVE = "knowledge/company_task_specialization/v4_fact_state/policy_records.jsonl"
V3_PROTOCOL_RELATIVE = "knowledge/company_task_specialization/v3_remedy_logic/protocol.json"
V4_PROTOCOL_RELATIVE = "knowledge/company_task_specialization/v4_fact_state/protocol.json"
KNOWLEDGE_RECORDS_RELATIVE = "knowledge/records.jsonl"
SPLIT_SEED = 42
TARGET_COUNTS = {"train": 24, "dev": 8, "test": 8}
_ASSIGNMENT_CACHE: dict[tuple[tuple[str, int], ...], dict[str, str]] = {}

V3_FAMILY_MEMBERS: dict[str, tuple[str, ...]] = {
    "v3-po-allof-5": ("CTS3-QUAL-001",),
    "v3-po-cond-1": ("CTS3-QUAL-002", "CTS3-QUAL-005", "CTS3-QUAL-006", "CTS3-QUAL-007"),
    "v3-po-allof-2": ("CTS3-QUAL-003", "CTS3-QUAL-004"),
    "v3-po-allof-4": ("CTS3-QUAL-008",),
    "v3-inv-anyof-2": (
        "CTS3-QUAL-009",
        "CTS3-QUAL-010",
        "CTS3-QUAL-011",
        "CTS3-QUAL-012",
        "CTS3-QUAL-013",
        "CTS3-QUAL-016",
    ),
    "v3-pay-cond-1": ("CTS3-QUAL-014",),
    "v3-date-allof-2": ("CTS3-QUAL-015",),
    "v3-multi-onb-inv": ("CTS3-QUAL-017", "CTS3-QUAL-018", "CTS3-QUAL-019"),
    "v3-multi-po-inv": ("CTS3-QUAL-020",),
    "v3-nest-sec-inv": ("CTS3-QUAL-021", "CTS3-QUAL-022", "CTS3-QUAL-023"),
    "v3-nest-sec-terms-inv": ("CTS3-QUAL-024",),
}

HIDDEN_CASE_KEYS = (
    "reference",
    "policy_logic",
    "expected_fact_state",
    "authoring",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parent_assets(root: Path) -> dict[str, Any]:
    """Load frozen v3/v4 assets without rewriting them."""
    v3_cases = read_jsonl(root / V3_CASES_RELATIVE)
    v4_cases = read_jsonl(root / V4_CASES_RELATIVE)
    v4_policy = read_jsonl(root / V4_POLICY_RELATIVE)
    if len(v3_cases) != 24:
        raise ValueError(f"expected 24 v3 cases, found {len(v3_cases)}")
    if len(v4_cases) != 16:
        raise ValueError(f"expected 16 v4 cases, found {len(v4_cases)}")
    records = {record.id: record for record in load_records(root / "knowledge")}
    policy_by_id = {str(row["id"]): row for row in v4_policy}
    return {
        "v3_cases": v3_cases,
        "v4_cases": v4_cases,
        "v4_policy": v4_policy,
        "v4_policy_by_id": policy_by_id,
        "records": records,
        "hashes": {
            V3_CASES_RELATIVE: file_sha256(root / V3_CASES_RELATIVE),
            V4_CASES_RELATIVE: file_sha256(root / V4_CASES_RELATIVE),
            V4_POLICY_RELATIVE: file_sha256(root / V4_POLICY_RELATIVE),
            V3_PROTOCOL_RELATIVE: file_sha256(root / V3_PROTOCOL_RELATIVE),
            V4_PROTOCOL_RELATIVE: file_sha256(root / V4_PROTOCOL_RELATIVE),
            KNOWLEDGE_RECORDS_RELATIVE: file_sha256(root / KNOWLEDGE_RECORDS_RELATIVE),
        },
    }


def family_catalog(
    v3_cases: Sequence[Mapping[str, Any]], v4_cases: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build family IDs that keep contrast pairs and shared policy families together."""
    v3_by_id = {str(case["case_id"]): case for case in v3_cases}
    expected_v3 = {case_id for members in V3_FAMILY_MEMBERS.values() for case_id in members}
    if expected_v3 != set(v3_by_id):
        raise ValueError("v3 family catalog does not cover the frozen case IDs")
    families: list[dict[str, Any]] = []
    for family_id, members in V3_FAMILY_MEMBERS.items():
        cases = [v3_by_id[case_id] for case_id in members]
        families.append(
            {
                "family_id": family_id,
                "source": "v3",
                "case_ids": list(members),
                "pair_ids": [],
                "tags": sorted({tag for case in cases for tag in case.get("evaluation_tags", [])}),
                "size": len(members),
            }
        )
    pairs: dict[str, list[str]] = {}
    for case in v4_cases:
        pairs.setdefault(str(case["pair_id"]), []).append(str(case["case_id"]))
    if len(pairs) != 8:
        raise ValueError(f"expected 8 v4 pairs, found {len(pairs)}")
    v4_by_id = {str(case["case_id"]): case for case in v4_cases}
    for pair_id, members in sorted(pairs.items()):
        if len(members) != 2:
            raise ValueError(f"{pair_id} is not a complete contrast pair")
        cases = [v4_by_id[case_id] for case_id in members]
        families.append(
            {
                "family_id": f"v4-{pair_id.lower()}",
                "source": "v4",
                "case_ids": members,
                "pair_ids": [pair_id],
                "tags": sorted({tag for case in cases for tag in case.get("evaluation_tags", [])}),
                "size": 2,
            }
        )
    if sum(family["size"] for family in families) != 40:
        raise ValueError("family catalog does not contain 40 cases")
    return {"families": families}


def assign_splits(
    families: Sequence[Mapping[str, Any]], *, seed: int = SPLIT_SEED
) -> dict[str, str]:
    """Assign whole families to train/dev/test after grouping. Seed is used only here."""
    indexed = list(families)
    cache_key = tuple((str(family["family_id"]), int(family["size"])) for family in indexed) + (
        ("__seed__", seed),
    )
    cached = _ASSIGNMENT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    sizes = [int(family["size"]) for family in indexed]
    n = len(indexed)
    test_masks = _masks_with_sum(sizes, TARGET_COUNTS["test"])
    candidates: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for test_mask in test_masks:
        rem_idx = [i for i in range(n) if not (test_mask & (1 << i))]
        rem_sizes = [sizes[i] for i in rem_idx]
        for dev_local in _masks_with_sum(rem_sizes, TARGET_COUNTS["dev"]):
            dev_idx = [rem_idx[j] for j in range(len(rem_idx)) if dev_local & (1 << j)]
            test_idx = [i for i in range(n) if test_mask & (1 << i)]
            train_idx = [i for i in rem_idx if i not in dev_idx]
            if sum(sizes[i] for i in train_idx) != TARGET_COUNTS["train"]:
                continue
            if not _has_required_coverage(indexed, test_idx, dev_idx, train_idx):
                continue
            candidates.append(
                (
                    tuple(sorted(indexed[i]["family_id"] for i in test_idx)),
                    tuple(sorted(indexed[i]["family_id"] for i in dev_idx)),
                    tuple(sorted(indexed[i]["family_id"] for i in train_idx)),
                )
            )
    if not candidates:
        raise RuntimeError("no leakage-resistant split satisfied the coverage constraints")
    candidates = sorted(set(candidates))
    chosen = candidates[random.Random(seed).randrange(len(candidates))]
    assignment: dict[str, str] = {}
    for family_id in chosen[0]:
        assignment[family_id] = "test"
    for family_id in chosen[1]:
        assignment[family_id] = "dev"
    for family_id in chosen[2]:
        assignment[family_id] = "train"
    _ASSIGNMENT_CACHE[cache_key] = dict(assignment)
    return assignment


def _masks_with_sum(sizes: Sequence[int], target: int) -> list[int]:
    found: list[int] = []
    n = len(sizes)
    for mask in range(1 << n):
        total = 0
        overflow = False
        remaining = mask
        index = 0
        while remaining:
            if remaining & 1:
                total += sizes[index]
                if total > target:
                    overflow = True
                    break
            remaining >>= 1
            index += 1
        if not overflow and total == target:
            found.append(mask)
    return found


def _has_required_coverage(
    families: Sequence[Mapping[str, Any]],
    test_idx: Sequence[int],
    dev_idx: Sequence[int],
    train_idx: Sequence[int],
) -> bool:
    test = [families[i] for i in test_idx]
    train = [families[i] for i in train_idx]
    if not any(family["source"] == "v4" for family in test):
        return False
    if not any(family["source"] == "v3" for family in test):
        return False
    if not any(
        "any_of" in family["tags"] or family["family_id"].startswith("v3-inv") for family in test
    ):
        return False
    if not any(family["source"] == "v4" for family in train):
        return False
    if not any(
        "any_of" in family["tags"] or "alternative_remedies" in family["tags"] for family in train
    ):
        return False
    if not any(
        "nested_logic" in family["tags"] or family["family_id"].startswith("v3-nest")
        for family in train
    ):
        return False
    del dev_idx
    return True


def compile_rows(root: Path) -> dict[str, Any]:
    assets = load_parent_assets(root)
    catalog = family_catalog(assets["v3_cases"], assets["v4_cases"])
    assignment = assign_splits(catalog["families"])
    cases = {str(case["case_id"]): {**case, "source_protocol": "v3"} for case in assets["v3_cases"]}
    cases.update(
        {str(case["case_id"]): {**case, "source_protocol": "v4"} for case in assets["v4_cases"]}
    )
    family_by_case: dict[str, dict[str, Any]] = {}
    for family in catalog["families"]:
        for case_id in family["case_ids"]:
            family_by_case[case_id] = family
    rows = []
    for case_id, case in sorted(cases.items()):
        family = family_by_case[case_id]
        split = assignment[family["family_id"]]
        prompt = render_prompt(root, case, assets)
        hidden_fragments = _hidden_fragments(case)
        for fragment in hidden_fragments:
            if fragment and fragment in prompt["question"]:
                raise ValueError(f"{case_id} leaked hidden reference text into the prompt")
        rows.append(
            {
                "case_id": case_id,
                "family_id": family["family_id"],
                "pair_id": case.get("pair_id"),
                "source_protocol": case["source_protocol"],
                "split": split,
                "evaluation_tags": list(case.get("evaluation_tags", [])),
                "request_item_ids": [
                    str(item["request_item_id"]) for item in case["request_items"]
                ],
                "source_record_ids": list(case["source_record_ids"]),
                "authorized_ids": authorized_ids(case),
                "row_sha256": sha256_json(
                    {
                        "case_id": case_id,
                        "family_id": family["family_id"],
                        "split": split,
                        "prompt": prompt,
                    }
                ),
                "prompt_sha256": sha256_text(prompt["question"]),
                "system_prompt_sha256": sha256_text(prompt["system_prompt"]),
            }
        )
    _assert_split_invariants(rows, catalog["families"], assignment)
    return {
        "rows": rows,
        "families": catalog["families"],
        "assignment": assignment,
        "parent_hashes": assets["hashes"],
        "tag_distribution": _tag_distribution(rows),
        "source_record_distribution": _source_distribution(rows),
        "pair_distribution": _pair_distribution(rows),
    }


def authorized_ids(case: Mapping[str, Any]) -> list[str]:
    values = [str(item) for item in case["source_record_ids"]]
    for state in case.get("operational_state") or []:
        values.append(str(state["state_id"]))
    return values


def render_prompt(
    root: Path, case: Mapping[str, Any], assets: Mapping[str, Any] | None = None
) -> dict[str, str]:
    loaded = assets or load_parent_assets(root)
    if case.get("source_protocol") == "v4" or "operational_state" in case:
        evidence = []
        for record_id in case["source_record_ids"]:
            record = loaded["v4_policy_by_id"].get(str(record_id))
            if record is None:
                raise ValueError(f"missing v4 policy record {record_id}")
            evidence.append(record)
        payload = {
            "evidence": evidence,
            "state_authority_guide": STATE_AUTHORITY_GUIDE,
            "operational_state": case["operational_state"],
            "request_items": case["request_items"],
        }
        system_prompt = AMENDED_SYSTEM_PROMPT
    else:
        evidence = []
        for record_id in case["source_record_ids"]:
            record = loaded["records"].get(str(record_id))
            if record is None:
                raise ValueError(f"missing knowledge record {record_id}")
            evidence.append(_source_dict(record))
        payload = {
            "evidence": evidence,
            "operational_facts": case["operational_facts"],
            "request_items": case["request_items"],
        }
        system_prompt = CHALLENGER_SYSTEM_PROMPT
    question = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {"system_prompt": system_prompt, "question": question}


def _hidden_fragments(case: Mapping[str, Any]) -> list[str]:
    """Hidden evaluator objects. Evidence-duplicated policy sentences are not hidden."""
    fragments: list[str] = []
    for key in HIDDEN_CASE_KEYS:
        value = case.get(key)
        if value is None:
            continue
        fragments.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return fragments


def _assert_split_invariants(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
) -> None:
    counts = Counter(row["split"] for row in rows)
    if dict(counts) != TARGET_COUNTS:
        raise ValueError(f"split counts {dict(counts)} != {TARGET_COUNTS}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs in compiled rows")
    by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    family_split: dict[str, set[str]] = {}
    pair_split: dict[str, set[str]] = {}
    for row in rows:
        by_split[str(row["split"])].add(str(row["case_id"]))
        family_split.setdefault(str(row["family_id"]), set()).add(str(row["split"]))
        if row.get("pair_id"):
            pair_split.setdefault(str(row["pair_id"]), set()).add(str(row["split"]))
    if (
        by_split["train"] & by_split["dev"]
        or by_split["train"] & by_split["test"]
        or by_split["dev"] & by_split["test"]
    ):
        raise ValueError("case overlap across splits")
    if any(len(values) != 1 for values in family_split.values()):
        raise ValueError("family leaked across splits")
    if any(len(values) != 1 for values in pair_split.values()):
        raise ValueError("v4 pair leaked across splits")
    for family in families:
        expected = assignment[family["family_id"]]
        for case_id in family["case_ids"]:
            row = next(item for item in rows if item["case_id"] == case_id)
            if row["split"] != expected:
                raise ValueError(f"{case_id} split drifted from family assignment")


def _tag_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = distribution.setdefault(str(row["split"]), {})
        for tag in row["evaluation_tags"]:
            bucket[str(tag)] = bucket.get(str(tag), 0) + 1
    return distribution


def _source_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = distribution.setdefault(str(row["split"]), {})
        for record_id in row["source_record_ids"]:
            bucket[str(record_id)] = bucket.get(str(record_id), 0) + 1
    return distribution


def _pair_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    distribution: dict[str, list[str]] = {}
    for row in rows:
        if row.get("pair_id"):
            distribution.setdefault(str(row["split"]), []).append(str(row["pair_id"]))
    return {key: sorted(set(values)) for key, values in distribution.items()}

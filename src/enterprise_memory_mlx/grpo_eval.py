"""Held-out BASE vs RFT evaluation for the GRPO pilot."""

from __future__ import annotations

import hashlib
import json
import secrets
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .grpo_data import load_parent_assets
from .grpo_protocol import PROTOCOL_ID, spec_by_case
from .utils import atomic_write_text

DECISIONS = [
    "proceed",
    "do_not_proceed",
    "needs_action",
    "needs_information",
    "refer_to_source",
]


def case_json_schema(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Legal structure and IDs only. Does not encode correct pairings."""
    source_enum = list(spec["authorized_ids"])
    request_enum = list(spec["request_item_ids"])
    action_node = {
        "type": "object",
        "additionalProperties": False,
        "required": ["node_type", "action", "source_record_id"],
        "properties": {
            "node_type": {"type": "string", "const": "action"},
            "action": {"type": "string", "minLength": 1},
            "source_record_id": {"type": "string", "enum": source_enum},
        },
    }
    group_node: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["node_type", "operator", "options"],
        "properties": {
            "node_type": {"type": "string", "const": "group"},
            "operator": {"type": "string", "enum": ["any_of", "all_of"]},
            "options": {
                "type": "array",
                "minItems": 2,
                "items": {"$ref": "#/$defs/node"},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assessments"],
        "properties": {
            "assessments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "request_item_id",
                        "decision",
                        "current_blockers",
                        "remedy",
                        "missing_information",
                        "exceptions",
                        "evidence",
                    ],
                    "properties": {
                        "request_item_id": {"type": "string", "enum": request_enum},
                        "decision": {"type": "string", "enum": DECISIONS},
                        "current_blockers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["condition", "source_record_id"],
                                "properties": {
                                    "condition": {"type": "string", "minLength": 1},
                                    "source_record_id": {"type": "string", "enum": source_enum},
                                },
                            },
                        },
                        "remedy": {"anyOf": [{"type": "null"}, {"$ref": "#/$defs/node"}]},
                        "missing_information": {"type": "array", "items": {"type": "string"}},
                        "exceptions": {"type": "array", "items": {"type": "string"}},
                        "evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["record_id", "claim"],
                                "properties": {
                                    "record_id": {"type": "string", "enum": source_enum},
                                    "claim": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                },
            }
        },
        "$defs": {"node": {"anyOf": [action_node, group_node]}},
    }


def summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rewards = [float(row["score"]["reward"]) for row in rows]
    full = sum(1 for row in rows if row["score"]["full_machine_success"])
    decision_items = 0
    decision_correct = 0
    for row in rows:
        for flag in row["score"]["decision_correctness"]:
            decision_items += 1
            decision_correct += int(bool(flag["correct"]))
    return {
        "cases": n,
        "mean_reward": round(sum(rewards) / n, 6) if n else 0.0,
        "full_machine_success": full,
        "exact_decision_accuracy": (
            round(decision_correct / decision_items, 6) if decision_items else 0.0
        ),
        "valid_outputs": sum(1 for row in rows if row["score"]["complete_valid_output"]),
        "provenance_hard_failures": sum(1 for row in rows if row["score"]["provenance_hard_fail"]),
        "unsafe_hard_failures": sum(1 for row in rows if row["score"]["unsafe_hard_fail"]),
    }


def compare_arms(
    base_rows: Sequence[Mapping[str, Any]], rft_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    base_by_id = {row["case_id"]: row for row in base_rows}
    rft_by_id = {row["case_id"]: row for row in rft_rows}
    wins = losses = ties = 0
    paired = []
    for case_id in sorted(base_by_id):
        base = base_by_id[case_id]["score"]
        rft = rft_by_id[case_id]["score"]
        if rft["reward"] > base["reward"]:
            outcome = "win"
            wins += 1
        elif rft["reward"] < base["reward"]:
            outcome = "loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1
        paired.append(
            {
                "case_id": case_id,
                "family_id": base_by_id[case_id].get("family_id"),
                "base_reward": base["reward"],
                "rft_reward": rft["reward"],
                "outcome": outcome,
            }
        )
    base_summary = summarize_arm(base_rows)
    rft_summary = summarize_arm(rft_rows)
    new_unsafe = max(0, rft_summary["unsafe_hard_failures"] - base_summary["unsafe_hard_failures"])
    gate = {
        "mean_reward_delta": round(rft_summary["mean_reward"] - base_summary["mean_reward"], 6),
        "full_success_delta": rft_summary["full_machine_success"]
        - base_summary["full_machine_success"],
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "provenance_did_not_increase": (
            rft_summary["provenance_hard_failures"] <= base_summary["provenance_hard_failures"]
        ),
        "new_unsafe_hard_failures": new_unsafe,
        "decision_accuracy_did_not_regress": (
            rft_summary["exact_decision_accuracy"] >= base_summary["exact_decision_accuracy"]
        ),
    }
    passed = (
        gate["mean_reward_delta"] >= 0.10
        and gate["full_success_delta"] >= 2
        and wins > losses
        and gate["provenance_did_not_increase"]
        and new_unsafe == 0
        and gate["decision_accuracy_did_not_regress"]
    )
    return {
        "base": base_summary,
        "rft": rft_summary,
        "paired": paired,
        "gate": gate,
        "status": (
            "rft_pilot_promising_semantic_review_required" if passed else "stop_rft_candidate"
        ),
    }


def write_review_packet(
    root: Path,
    *,
    evaluation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Blind BASE/RFT TEST outputs. Does not set human_attested."""
    comparison = json.loads((evaluation_dir / "comparison.json").read_text(encoding="utf-8"))
    specs = spec_by_case(root)
    assets = load_parent_assets(root)
    cases = {str(case["case_id"]): case for case in [*assets["v3_cases"], *assets["v4_cases"]]}
    rows = []
    mapping = []
    for surface in ("primary_structured", "secondary_unconstrained"):
        for arm in ("base", "rft"):
            for item in comparison[surface]["rows"][arm]:
                case = cases[item["case_id"]]
                review_id = f"GRPO-{secrets.token_hex(8).upper()}"
                rows.append(
                    {
                        "review_id": review_id,
                        "surface": surface,
                        "operational_facts": case.get("operational_facts"),
                        "operational_state": case.get("operational_state"),
                        "request_items": case["request_items"],
                        "source_record_ids": case["source_record_ids"],
                        "candidate_answer": item.get("output"),
                        "source_ids": specs[item["case_id"]]["authorized_ids"],
                    }
                )
                mapping.append(
                    {
                        "review_id": review_id,
                        "arm": arm,
                        "case_id": item["case_id"],
                        "surface": surface,
                        "reward": item["score"]["reward"],
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = output_dir / "grpo-test-review.zip"
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "REVIEW_INSTRUCTIONS.md",
            (
                "Assess unsupported evidence claims, invented requirements, missing work vs "
                "missing evidence, authority/referral correctness, unsafe or contradictory "
                "actions, and internal consistency. Do not infer arm identity.\n"
            ),
        )
        archive.writestr(
            "review_cases.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
    mapping_path = output_dir / "private" / "review_id_map.json"
    atomic_write_text(
        mapping_path, json.dumps({"mapping": mapping}, indent=2, sort_keys=True) + "\n"
    )
    return {
        "packet": str(packet),
        "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        "mapping": str(mapping_path),
        "semantic_review": "pending",
        "human_attested": False,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": PROTOCOL_ID,
    }

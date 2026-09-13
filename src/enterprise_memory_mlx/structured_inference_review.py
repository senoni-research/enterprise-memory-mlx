"""Blinded local review packet for the ten measured structured-inference outputs.

This is not prepare_content_review(). That function selects historical
unconstrained after-answers. This packager reads only the measured run.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .learning_mechanics import load_approval, sha256_file, verify_bound_pairs, write_json
from .structured_inference import CASE_ORDER, PROTOCOL_ID
from .utils import atomic_write_text, sha256_json, sha256_text

MEASURED_RUN_ID = "run-20260912T112818Z"
SETUP_FAILURE_RUN_ID = "run-20260912T112732Z"
ARMS = (
    ("arm-a-unadapted", "arm-a-unadapted"),
    ("arm-b-adapter", "arm-b-adapter"),
)
PACKET_PURPOSE = "structured-inference-measured-ten-output-blinded-review/v1"
REFERENCE_BRIEF = (
    "No reference review is provided for this packet.\n\n"
    "Do not assume an arm, adapter status, prior score, timing, or approved label.\n\n"
    "Assess the candidate against the frozen model-visible question and the local "
    "evidence paths listed in governed source records:\n"
    "1. evidence support;\n"
    "2. omitted or invented requirements;\n"
    "3. missing-work versus missing-evidence distinctions;\n"
    "4. authority and referral handling;\n"
    "5. harmful or contradictory next actions;\n"
    "6. consistency between labels, disposition and rationale.\n\n"
    "Do not infer action safety from label agreement. Do not export private evidence."
)
REVIEW_INSTRUCTIONS = """# Measured structured-inference content review

Authorized local reviewer only.

This packet contains ten complete raw answers from the measured constrained
run. Arm names, adapter status, scores, timing and previous judgments are
omitted from this view.

Assess each candidate on:

- evidence support;
- omitted or invented requirements;
- missing-work versus missing-evidence distinctions;
- authority and referral handling;
- harmful or contradictory next actions;
- consistency between labels, disposition and rationale.

Use the local evidence files named in governed source records. Keep those
files on this machine. Do not send private evidence to an external service.

Do not edit the candidate answers. There is no reference answer to match.
Do not infer safety from any later unblinding of scores.

Open with the existing local review UI:

```
emmlx review --packet <this-zip> --mapping <private-review_id_map.json> --reviewer <id>
```
"""
BLINDED_FORBIDDEN = {
    "arm",
    "arm_id",
    "adapter_path",
    "case_id",
    "score",
    "scores",
    "label_matches",
    "complete_valid",
    "disposition_agrees",
    "elapsed_seconds",
    "peak_memory_gb",
    "generation_status",
    "approved_labels",
    "model_labels",
    "approved_disposition",
    "model_disposition",
}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def measured_run_dir(root: Path) -> Path:
    return root / "artifacts" / "qwen4b-structured-inference-v1" / MEASURED_RUN_ID


def output_path(root: Path, arm_id: str, case_id: str) -> Path:
    return measured_run_dir(root) / arm_id / "outputs" / f"{case_id}.json"


def collect_measured_outputs(root: Path) -> list[dict[str, Any]]:
    approval = load_approval(root)
    verified = {item["case_id"]: item for item in verify_bound_pairs(root, approval)}
    rows = []
    for arm_id, _arm_dir in ARMS:
        for case_id in CASE_ORDER:
            path = output_path(root, arm_id, case_id)
            if not path.is_file():
                raise FileNotFoundError(f"measured output missing: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw = payload.get("raw_output")
            if not isinstance(raw, str) or not raw:
                raise RuntimeError(f"{arm_id} {case_id} has no raw_output")
            item = verified[case_id]
            source = json.loads(Path(item["input_path"]).read_text(encoding="utf-8"))
            rows.append(
                {
                    "arm_id": arm_id,
                    "case_id": case_id,
                    "mapping_id": f"{arm_id}/{case_id}",
                    "source_output_path": str(path.relative_to(root)),
                    "source_output_file_sha256": sha256_file(path),
                    "raw_output_sha256": sha256_text(raw),
                    "raw_output": raw,
                    "input_sha256": item["input_sha256"],
                    "target_sha256": item["target_sha256"],
                    "user_prompt": source["user_prompt"],
                    "source_index": list(source["source_index"]),
                }
            )
    if len(rows) != 10:
        raise RuntimeError(f"expected ten measured outputs, found {len(rows)}")
    return rows


def _source_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = {}
    for row in rows:
        for entry in row["source_index"]:
            record_id = f"{row['case_id']}/{entry['source_id']}"
            records[record_id] = {
                "id": record_id,
                "title": str(entry.get("path") or entry["source_id"]),
                "statement": (
                    f"Local evidence only. source_id={entry['source_id']} "
                    f"sha256={entry.get('sha256')} "
                    f"line_ranges={entry.get('line_ranges')} "
                    f"kind={entry.get('kind')}. Do not export."
                ),
            }
    return [records[key] for key in sorted(records)]


def prepare_measured_constrained_review(
    root: Path,
    *,
    salt: bytes | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Create a new blinded packet from the ten measured constrained outputs."""
    rows = collect_measured_outputs(root)
    private_salt = salt if salt is not None else secrets.token_bytes(32)
    if len(private_salt) < 16:
        raise RuntimeError("review salt must contain at least 16 bytes")
    source_catalog = {
        "measured_run_id": MEASURED_RUN_ID,
        "outputs": [
            {
                "mapping_id": row["mapping_id"],
                "source_output_path": row["source_output_path"],
                "source_output_file_sha256": row["source_output_file_sha256"],
                "raw_output_sha256": row["raw_output_sha256"],
                "input_sha256": row["input_sha256"],
                "target_sha256": row["target_sha256"],
            }
            for row in rows
        ],
    }
    source_artifact_sha256 = sha256_json(source_catalog)
    blinded: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for row in rows:
        digest = hmac.new(
            private_salt,
            row["mapping_id"].encode("utf-8"),
            hashlib.sha256,
        )
        review_id = f"SIR-{digest.hexdigest()[:16].upper()}"
        source_ids = [
            f"{row['case_id']}/{entry['source_id']}" for entry in row["source_index"]
        ]
        blinded_row = {
            "review_id": review_id,
            "question": row["user_prompt"],
            "reference_answer": REFERENCE_BRIEF,
            "candidate_answer": row["raw_output"],
            "source_record_ids": source_ids,
        }
        leaked = BLINDED_FORBIDDEN & blinded_row.keys()
        if leaked:
            raise RuntimeError(f"blinded row leaked {sorted(leaked)}")
        blinded.append(blinded_row)
        mapping_rows.append(
            {
                "review_id": review_id,
                "case_id": row["mapping_id"],
                "measured_case_id": row["case_id"],
                "arm_id": row["arm_id"],
                "source_output_path": row["source_output_path"],
                "source_output_file_sha256": row["source_output_file_sha256"],
                "raw_output_sha256": row["raw_output_sha256"],
                "input_sha256": row["input_sha256"],
                "target_sha256": row["target_sha256"],
            }
        )
    blinded.sort(key=lambda item: sha256_text(str(item["review_id"])))
    mapping_rows.sort(key=lambda item: str(item["review_id"]))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_root = output_root or (
        root
        / "artifacts"
        / "review-packets"
        / "qwen4b-structured-inference-v1-measured-ten-output"
        / stamp
    )
    if packet_root.exists():
        raise FileExistsError(f"Refusing to overwrite review packet: {packet_root}")
    private_dir = packet_root / "private"
    private_dir.mkdir(parents=True, exist_ok=False)
    mapping_payload = {
        "schema_version": 1,
        "packet_purpose": PACKET_PURPOSE,
        "protocol_id": PROTOCOL_ID,
        "source_artifact_sha256": source_artifact_sha256,
        "source_catalog": source_catalog,
        "salt_hex": private_salt.hex(),
        "mapping": mapping_rows,
        "hidden_from_initial_reviewer_view": [
            "arm_id",
            "adapter_status",
            "scores",
            "timing",
            "previous_judgments",
        ],
    }
    mapping_text = json.dumps(mapping_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    mapping_path = private_dir / "review_id_map.json"
    atomic_write_text(mapping_path, mapping_text)
    sources = _source_records(rows)
    members = {
        "REVIEW_INSTRUCTIONS.md": REVIEW_INSTRUCTIONS.encode("utf-8"),
        "review_cases.jsonl": _jsonl_bytes(blinded),
        "source_records.jsonl": _jsonl_bytes(sources),
    }
    packet_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": PACKET_PURPOSE,
        "status": "human_review_pending",
        "semantic_assessment": "pending",
        "promotion_eligible": False,
        "source_candidate": {
            "artifact_type": "structured-inference-measured-outputs",
            "primary_sha256": source_artifact_sha256,
            "measured_run_id": MEASURED_RUN_ID,
        },
        "private_mapping_sha256": sha256_text(mapping_text),
        "case_count": len(blinded),
        "blinding": {
            "arm_excluded": True,
            "adapter_status_excluded": True,
            "scores_excluded": True,
            "timing_excluded": True,
            "previous_judgments_excluded": True,
            "approved_labels_excluded": True,
            "private_mapping_in_zip": False,
            "private_evidence_file_bytes_excluded": True,
        },
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(members.items())
        },
    }
    manifest_bytes = (
        json.dumps(packet_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    package_name = "structured-inference-measured-ten-output"
    packet_path = packet_root / f"{package_name}.zip"
    temporary = packet_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = f"{package_name}/"
        for name, content in sorted(members.items()):
            archive.writestr(prefix + name, content)
        archive.writestr(prefix + "packet_manifest.json", manifest_bytes)
    temporary.replace(packet_path)
    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    status = {
        "protocol_id": PROTOCOL_ID,
        "packet_purpose": PACKET_PURPOSE,
        "semantic_assessment": "pending",
        "blocks_full_task_success": True,
        "human_review_fabricated": False,
        "external_service_used": False,
        "measured_answers_edited": False,
        "case_count": 10,
        "packet_path": _rel(packet_path),
        "mapping_path": _rel(mapping_path),
        "packet_sha256": sha256_file(packet_path),
        "mapping_sha256": sha256_text(mapping_text),
        "source_artifact_sha256": source_artifact_sha256,
    }
    write_json(packet_root / "status.json", status)
    return status

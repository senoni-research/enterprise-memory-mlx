"""Local blinded human-review UI and project status dashboard."""

# Embedded HTML/CSS/JavaScript is intentionally kept dependency-free.
# ruff: noqa: E501

from __future__ import annotations

import getpass
import hashlib
import json
import secrets
import threading
import webbrowser
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .legacy_guard import LEGACY_COMMANDS
from .split_contract import verify_frozen_assets
from .utils import atomic_write_text, slugify

LEGAL_SCORES = (0.0, 0.5, 1.0)
LEGAL_CONFIDENCE = ("high", "medium", "low")
IDENTITY_VERIFICATION = "asserted_only_not_authenticated"
DEFAULT_PACKET = Path(
    "artifacts/review-packets/judge-calibration-v3-model-review.zip"
)
DEFAULT_MAPPING = Path(
    "artifacts/review-packets/judge-calibration-v3-private/review_id_map.json"
)
DEFAULT_STATE_ROOT = Path("artifacts/human-reviews/judge-calibration-v3")


class ReviewDataError(ValueError):
    """The blinded packet, private mapping, or review state is invalid."""


@dataclass(frozen=True)
class ReviewDecision:
    review_id: str
    reviewer_id: str
    score: float
    reason: str
    confidence: str
    needs_human_attention: bool
    reviewed_at: str
    human_attested: bool
    os_user: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReviewDecision:
        score = value.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ReviewDataError("score must be 0.0, 0.5, or 1.0")
        numeric_score = float(score)
        if numeric_score not in LEGAL_SCORES:
            raise ReviewDataError("score must be 0.0, 0.5, or 1.0")
        confidence = str(value.get("confidence", "")).strip().lower()
        if confidence not in LEGAL_CONFIDENCE:
            raise ReviewDataError("confidence must be high, medium, or low")
        reason = str(value.get("reason", "")).strip()
        if not reason:
            raise ReviewDataError("a case-specific reason is required")
        reviewer_id = str(value.get("reviewer_id", "")).strip()
        review_id = str(value.get("review_id", "")).strip()
        reviewed_at = str(value.get("reviewed_at", "")).strip()
        if not reviewer_id or not review_id or not reviewed_at:
            raise ReviewDataError("review_id, reviewer_id, and reviewed_at are required")
        os_user = str(value.get("os_user", "")).strip()
        if not os_user:
            raise ReviewDataError("the invoking OS user must be recorded")
        human_attested = value.get("human_attested") is True
        if not human_attested:
            raise ReviewDataError("human reviewer attestation is required")
        return cls(
            review_id=review_id,
            reviewer_id=reviewer_id,
            score=numeric_score,
            reason=reason,
            confidence=confidence,
            needs_human_attention=value.get("needs_human_attention") is True,
            reviewed_at=reviewed_at,
            human_attested=True,
            os_user=os_user,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "reviewer_id": self.reviewer_id,
            "reviewer_kind": "human",
            "identity_verification": IDENTITY_VERIFICATION,
            "os_user": self.os_user,
            "score": self.score,
            "reason": self.reason,
            "confidence": self.confidence,
            "needs_human_attention": self.needs_human_attention,
            "reviewed_at": self.reviewed_at,
            "human_attested": self.human_attested,
        }


class ReviewStore:
    """Verified blinded cases plus one human reviewer's persistent state."""

    def __init__(
        self,
        *,
        root: Path,
        packet_path: Path,
        mapping_path: Path,
        reviewer_id: str,
        state_root: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.packet_path = packet_path.resolve()
        self.mapping_path = mapping_path.resolve()
        self.reviewer_id = reviewer_id.strip()
        if not self.reviewer_id:
            raise ReviewDataError("reviewer identity cannot be empty")
        selected_state_root = state_root or (self.root / DEFAULT_STATE_ROOT)
        self.state_root = selected_state_root.resolve()
        self.state_path = (
            self.state_root / "reviewers" / f"{slugify(self.reviewer_id)}.jsonl"
        )
        self.export_path = (
            self.state_root / "overlays" / f"{slugify(self.reviewer_id)}.jsonl"
        )
        self.export_manifest_path = self.export_path.with_suffix(".manifest.json")
        self._lock = threading.Lock()
        (
            self.cases,
            self.source_records,
            self.packet_manifest,
        ) = _load_and_verify_packet(self.packet_path)
        self._case_by_id = {str(row["review_id"]): row for row in self.cases}
        self._mapping = _load_and_verify_mapping(
            self.mapping_path,
            set(self._case_by_id),
            self.packet_manifest,
        )
        self.decisions = self._load_state()

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def completed(self) -> int:
        return len(self.decisions)

    @property
    def remaining(self) -> int:
        return self.total - self.completed

    def overview(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "reviewer_kind": "human",
            "total": self.total,
            "completed": self.completed,
            "remaining": self.remaining,
            "complete": self.remaining == 0,
            "packet_sha256": _sha256_file(self.packet_path),
            "state_path": str(self.state_path),
            "export_path": str(self.export_path),
            "case_ids": [str(row["review_id"]) for row in self.cases],
            "completed_ids": sorted(self.decisions),
        }

    def case(self, review_id: str) -> dict[str, Any]:
        try:
            row = self._case_by_id[review_id]
        except KeyError as exc:
            raise ReviewDataError(f"unknown review_id: {review_id}") from exc
        source_by_id = {str(item["id"]): item for item in self.source_records}
        sources = [
            source_by_id[source_id]
            for source_id in row.get("source_record_ids", [])
            if source_id in source_by_id
        ]
        decision = self.decisions.get(review_id)
        return {
            "review_id": review_id,
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "candidate_answer": row["candidate_answer"],
            "source_records": sources,
            "decision": decision.to_dict() if decision else None,
        }

    def save(
        self,
        *,
        review_id: str,
        score: float,
        reason: str,
        confidence: str,
        needs_human_attention: bool,
        human_attested: bool,
    ) -> ReviewDecision:
        if review_id not in self._case_by_id:
            raise ReviewDataError(f"unknown review_id: {review_id}")
        decision = ReviewDecision.from_dict(
            {
                "review_id": review_id,
                "reviewer_id": self.reviewer_id,
                "score": score,
                "reason": reason,
                "confidence": confidence,
                "needs_human_attention": needs_human_attention,
                "reviewed_at": datetime.now(UTC).isoformat(),
                "human_attested": human_attested,
                "os_user": getpass.getuser(),
            }
        )
        with self._lock:
            self.decisions[review_id] = decision
            self._write_state()
        return decision

    def export_overlay(self) -> tuple[Path, Path]:
        if self.remaining:
            raise ReviewDataError(
                f"review is incomplete: {self.remaining} of {self.total} cases remain"
            )
        rows = []
        for case in self.cases:
            review_id = str(case["review_id"])
            decision = self.decisions[review_id]
            rows.append(
                {
                    "case_id": self._mapping[review_id],
                    "human_semantic_score": decision.score,
                    "human_reason": decision.reason,
                    "human_reviewers": [self.reviewer_id],
                    "human_approved": False,
                    "reviewer_kind": "human",
                    "identity_verification": IDENTITY_VERIFICATION,
                    "os_user": decision.os_user,
                    "confidence": decision.confidence,
                    "needs_human_attention": decision.needs_human_attention,
                    "reviewed_at": decision.reviewed_at,
                }
            )
        content = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        atomic_write_text(self.export_path, content)
        manifest = {
            "schema_version": 1,
            "status": "single_human_review_complete_not_adjudicated",
            "reviewer_id": self.reviewer_id,
            "reviewer_kind": "human",
            "identity_verification": IDENTITY_VERIFICATION,
            "os_users": sorted({decision.os_user for decision in self.decisions.values()}),
            "case_count": len(rows),
            "source_packet_sha256": _sha256_file(self.packet_path),
            "private_mapping_sha256": _sha256_file(self.mapping_path),
            "overlay_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "human_approved": False,
            "requires_second_review": True,
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_text(
            self.export_manifest_path,
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        return self.export_path, self.export_manifest_path

    def _load_state(self) -> dict[str, ReviewDecision]:
        if not self.state_path.exists():
            return {}
        decisions: dict[str, ReviewDecision] = {}
        for line_number, line in enumerate(
            self.state_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                decision = ReviewDecision.from_dict(json.loads(line))
            except (json.JSONDecodeError, ReviewDataError) as exc:
                raise ReviewDataError(
                    f"invalid review state at line {line_number}: {exc}"
                ) from exc
            if decision.reviewer_id != self.reviewer_id:
                raise ReviewDataError("review state belongs to another reviewer")
            if decision.review_id not in self._case_by_id:
                raise ReviewDataError(
                    f"review state contains unknown ID: {decision.review_id}"
                )
            decisions[decision.review_id] = decision
        return decisions

    def _write_state(self) -> None:
        rows = [
            self.decisions[review_id].to_dict()
            for review_id in sorted(self.decisions)
        ]
        content = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        atomic_write_text(self.state_path, content)


def project_status(root: Path, store: ReviewStore) -> dict[str, Any]:
    decision_path = (
        root / "knowledge" / "operating_points" / "bm25" / "v1-no-feasible.json"
    )
    bm25_status = "decision_missing"
    if decision_path.is_file():
        try:
            bm25_status = str(
                json.loads(decision_path.read_text(encoding="utf-8")).get(
                    "status", "invalid_decision"
                )
            )
        except json.JSONDecodeError:
            bm25_status = "invalid_decision"
    frozen_problems = verify_frozen_assets(root / "knowledge" / "eval_frozen")
    temporal_v2_problems = verify_frozen_assets(
        root / "knowledge" / "eval_frozen" / "v2"
    )
    research_bm25_path = (
        root
        / "knowledge"
        / "operating_points"
        / "bm25"
        / "v1-research-tradeoff.json"
    )
    smoke_run = (
        root
        / "artifacts"
        / "acquisition"
        / "runs"
        / "smoke_non_promotable-v2-micro-iterations-r16-e24-s42.json"
    )
    grading_reports = sorted(
        (root / "artifacts" / "grading").glob("deterministic-grading-*.json")
    )
    return {
        "overall": "NOT READY FOR CONFIRMATORY TRAINING",
        "training": "confirmatory_blocked_smoke_only",
        "promotion": "blocked",
        "legacy_commands": {
            "status": "blocked",
            "commands": sorted(LEGACY_COMMANDS),
        },
        "frozen_evaluation": {
            "status": "verified" if not frozen_problems else "invalid",
            "problems": frozen_problems,
            "supersession_v1": "preserved_not_valid_for_temporal_claims",
            "supersession_v2": (
                "verified_date_controlled"
                if not temporal_v2_problems
                else "invalid"
            ),
            "temporal_v2_problems": temporal_v2_problems,
        },
        "benchmark": {
            "default_arms": ["base", "full_context", "oracle"],
            "bm25": bm25_status,
            "bm25_research_control": (
                "experimental_non_promotable"
                if research_bm25_path.is_file()
                else "not_available"
            ),
            "answer_grading": (
                "deterministic_only_semantic_pending"
                if grading_reports
                else "not_run"
            ),
        },
        "acquisition_smoke": {
            "status": "trained_non_promotable" if smoke_run.is_file() else "not_run",
            "run_manifest": str(smoke_run) if smoke_run.is_file() else None,
            "latest_grading": (
                str(grading_reports[-1]) if grading_reports else None
            ),
        },
        "graders": {
            "strict": "integrated_deterministic_only",
            "provenance": "integrated_deterministic_only",
            "semantic_harness": "fake_backend_only_no_certified_judges",
        },
        "human_review": store.overview(),
    }


class _ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(_render_html(self.server.nonce))
            return
        if parsed.path == "/api/state":
            self._json(self.server.store.overview())
            return
        if parsed.path == "/api/status":
            self._json(project_status(self.server.root, self.server.store))
            return
        if parsed.path == "/api/case":
            review_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                self._json(self.server.store.case(review_id))
            except ReviewDataError as exc:
                self._error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("X-Review-Nonce") != self.server.nonce:
            self._error(HTTPStatus.FORBIDDEN, "invalid session nonce")
            return
        try:
            payload = self._read_json()
            if self.path == "/api/review":
                decision = self.server.store.save(
                    review_id=str(payload.get("review_id", "")),
                    score=payload.get("score"),
                    reason=str(payload.get("reason", "")),
                    confidence=str(payload.get("confidence", "")),
                    needs_human_attention=payload.get("needs_human_attention")
                    is True,
                    human_attested=payload.get("human_attested") is True,
                )
                self._json(
                    {
                        "decision": decision.to_dict(),
                        "progress": self.server.store.overview(),
                    }
                )
                return
            if self.path == "/api/export":
                overlay, manifest = self.server.store.export_overlay()
                self._json(
                    {
                        "overlay": str(overlay),
                        "manifest": str(manifest),
                        "overlay_sha256": _sha256_file(overlay),
                    }
                )
                return
        except (ReviewDataError, json.JSONDecodeError, TypeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ReviewDataError("invalid request body length")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ReviewDataError("request body must be a JSON object")
        return value

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._security_headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, value: str) -> None:
        body = value.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _security_headers(self, content_type: str, content_length: int) -> None:
        nonce = self.server.nonce
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; "
            f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
            "connect-src 'self'; form-action 'none'; frame-ancestors 'none'",
        )


class ReviewHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        *,
        root: Path,
        store: ReviewStore,
        nonce: str,
    ) -> None:
        super().__init__(address, _ReviewHandler)
        self.root = root
        self.store = store
        self.nonce = nonce


def create_server(
    *,
    root: Path,
    packet_path: Path,
    mapping_path: Path,
    reviewer_id: str,
    port: int,
    state_root: Path | None = None,
) -> ReviewHTTPServer:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
        raise ValueError("port must be an integer from 0 to 65535")
    store = ReviewStore(
        root=root,
        packet_path=packet_path,
        mapping_path=mapping_path,
        reviewer_id=reviewer_id,
        state_root=state_root,
    )
    return ReviewHTTPServer(
        ("127.0.0.1", port),
        root=root.resolve(),
        store=store,
        nonce=secrets.token_urlsafe(24),
    )


def serve_review_ui(
    *,
    root: Path,
    packet_path: Path,
    mapping_path: Path,
    reviewer_id: str,
    port: int = 8765,
    open_browser: bool = True,
    state_root: Path | None = None,
) -> None:
    server = create_server(
        root=root,
        packet_path=packet_path,
        mapping_path=mapping_path,
        reviewer_id=reviewer_id,
        port=port,
        state_root=state_root,
    )
    host, actual_port = server.server_address
    url = f"http://{host}:{actual_port}/"
    print(f"Blinded review UI: {url}")
    print("Press Ctrl-C to stop. Progress is saved after every decision.")
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _load_and_verify_packet(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"review packet not found: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest_name = _unique_suffix(names, "/packet_manifest.json")
        prefix = manifest_name.removesuffix("packet_manifest.json")
        manifest = json.loads(archive.read(manifest_name))
        recorded_files = manifest.get("files")
        if not isinstance(recorded_files, dict):
            raise ReviewDataError("packet manifest has no file hashes")
        for relative, expected in recorded_files.items():
            member = prefix + relative
            if member not in names:
                raise ReviewDataError(f"packet member missing: {relative}")
            actual = hashlib.sha256(archive.read(member)).hexdigest()
            if actual != expected:
                raise ReviewDataError(f"packet member hash mismatch: {relative}")
        cases = _parse_jsonl(
            archive.read(prefix + "review_cases.jsonl"), "review_cases.jsonl"
        )
        sources = _parse_jsonl(
            archive.read(prefix + "source_records.jsonl"), "source_records.jsonl"
        )
    if len(cases) != manifest.get("case_count"):
        raise ReviewDataError("packet case count does not match manifest")
    review_ids = [str(row.get("review_id", "")) for row in cases]
    if not all(review_ids) or len(set(review_ids)) != len(review_ids):
        raise ReviewDataError("packet review IDs are missing or duplicated")
    forbidden = {
        "arm",
        "question_id",
        "suite",
        "case_id",
        "question_family_id",
        "generation_status",
        "retrieval_label",
        "selected_record_ids",
        "source_uris",
        "context_action",
        "context_hash",
        "deterministic_status",
        "deterministic_score",
        "strict",
        "provenance",
        "reasons",
        "proposed_semantic_score",
        "proposed_reason",
        "error_categories",
        "critical_slot_observed_outcome",
        "provenance_observed_outcome",
        "certification_stratum",
        "case_family_id",
    }
    if any(forbidden & row.keys() for row in cases):
        raise ReviewDataError("packet exposes author-side or original-ID fields")
    return cases, sources, manifest


def _load_and_verify_mapping(
    path: Path,
    review_ids: set[str],
    packet_manifest: dict[str, Any],
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"private review mapping not found: {path}")
    mapping_bytes = path.read_bytes()
    payload = json.loads(mapping_bytes)
    expected_mapping_hash = packet_manifest.get("private_mapping_sha256")
    if (
        expected_mapping_hash is not None
        and hashlib.sha256(mapping_bytes).hexdigest() != expected_mapping_hash
    ):
        raise ReviewDataError("private mapping hash does not match the packet")
    source_candidate = packet_manifest.get("source_candidate", {})
    if not isinstance(source_candidate, dict):
        raise ReviewDataError("packet source candidate is invalid")
    source_hash = source_candidate.get("primary_sha256") or source_candidate.get(
        "cases_sha256"
    )
    mapping_source_hash = payload.get("source_artifact_sha256") or payload.get(
        "source_cases_sha256"
    )
    if mapping_source_hash != source_hash:
        raise ReviewDataError("private mapping belongs to another source candidate")
    mapping_rows = payload.get("mapping")
    if not isinstance(mapping_rows, list):
        raise ReviewDataError("private mapping has no rows")
    mapping = {
        str(row.get("review_id", "")): str(row.get("case_id", ""))
        for row in mapping_rows
        if isinstance(row, dict)
    }
    if set(mapping) != review_ids or any(not case_id for case_id in mapping.values()):
        raise ReviewDataError("private mapping IDs do not match the blinded packet")
    if len(set(mapping.values())) != len(mapping):
        raise ReviewDataError("private mapping contains duplicate original case IDs")
    return mapping


def _unique_suffix(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ReviewDataError(f"expected one packet member ending with {suffix}")
    return matches[0]


def _parse_jsonl(data: bytes, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        data.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReviewDataError(f"{label}:{line_number} is not an object")
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_html(nonce: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enterprise Memory Review</title>
<style nonce="{nonce}">
:root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: Canvas; color: CanvasText; }}
header {{ display:flex; justify-content:space-between; align-items:center; padding:16px 24px; border-bottom:1px solid GrayText; }}
h1 {{ font-size:20px; margin:0; }} h2 {{ font-size:17px; }} h3 {{ font-size:14px; }}
button, select, textarea {{ font:inherit; }}
button {{ padding:8px 12px; border:1px solid GrayText; border-radius:6px; background:Canvas; color:CanvasText; cursor:pointer; }}
button.primary {{ background:Highlight; color:HighlightText; border-color:Highlight; }}
button:disabled {{ opacity:.45; cursor:not-allowed; }}
nav {{ display:flex; gap:8px; }} main {{ max-width:1100px; margin:0 auto; padding:24px; }}
.hidden {{ display:none !important; }} .row {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
.space {{ justify-content:space-between; }} .muted {{ color:GrayText; }} .danger {{ color:#c33; font-weight:700; }}
.panel {{ border:1px solid GrayText; border-radius:8px; padding:16px; margin:14px 0; }}
.answer {{ white-space:pre-wrap; line-height:1.5; }}
.score {{ min-width:82px; }} .score.selected {{ outline:3px solid Highlight; }}
textarea {{ width:100%; min-height:90px; box-sizing:border-box; padding:10px; }}
.progress {{ height:8px; background:ButtonFace; border-radius:4px; overflow:hidden; }}
.progress > div {{ height:100%; background:Highlight; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px; }}
.status {{ border-left:4px solid GrayText; padding-left:12px; }}
code {{ overflow-wrap:anywhere; }} label {{ display:flex; gap:7px; align-items:center; }}
</style>
</head>
<body>
<header>
  <div><h1>Enterprise Memory</h1><span class="danger">CONFIRMATORY TRAINING BLOCKED</span></div>
  <nav><button id="reviewTab" class="primary">Human review</button><button id="statusTab">Status</button></nav>
</header>
<main>
<section id="reviewView">
  <div class="row space"><div><strong id="progressText">Loading…</strong></div><div class="row"><button id="prev">Previous</button><button id="next">Next</button></div></div>
  <div class="progress"><div id="progressBar"></div></div>
  <div class="panel"><h3>Question</h3><div id="question" class="answer"></div></div>
  <div class="panel"><h3>Reference answer</h3><div id="reference" class="answer"></div></div>
  <div class="panel"><h3>Candidate answer</h3><div id="candidate" class="answer"></div></div>
  <details class="panel"><summary>Governed source records</summary><div id="sources"></div></details>
  <div class="panel">
    <h3>Your independent judgment</h3>
    <div class="row">
      <button class="score" data-score="1">1.0 Correct</button>
      <button class="score" data-score="0.5">0.5 Partial</button>
      <button class="score" data-score="0">0.0 Incorrect</button>
      <select id="confidence"><option value="">Confidence…</option><option>high</option><option>medium</option><option>low</option></select>
      <label><input type="checkbox" id="attention"> Needs additional human attention</label>
    </div>
    <p><textarea id="reason" placeholder="Case-specific reason (required)"></textarea></p>
    <label><input type="checkbox" id="attest"> I am a human and made this decision myself; model labels were not shown.</label>
    <p class="row space"><span id="message" class="muted"></span><button id="save" class="primary">Save and next</button></p>
  </div>
  <button id="export" disabled>Export completed human overlay</button>
</section>
<section id="statusView" class="hidden">
  <h2>Programme status</h2>
  <div id="statusGrid" class="grid"></div>
</section>
</main>
<script nonce="{nonce}">
const nonce={json.dumps(nonce)};
let state=null,index=0,current=null,score=null;
const $=id=>document.getElementById(id);
async function api(path,options={{}}) {{
  options.headers={{'Content-Type':'application/json','X-Review-Nonce':nonce,...(options.headers||{{}})}};
  const r=await fetch(path,options); const data=await r.json();
  if(!r.ok) throw new Error(data.error||'Request failed'); return data;
}}
function esc(s) {{ const d=document.createElement('div'); d.textContent=s??''; return d.innerHTML; }}
async function loadState() {{
  state=await api('/api/state');
  const firstOpen=state.case_ids.findIndex(id=>!state.completed_ids.includes(id));
  index=firstOpen>=0?firstOpen:0; await loadCase();
}}
async function loadCase() {{
  if(!state.case_ids.length)return;
  current=await api('/api/case?id='+encodeURIComponent(state.case_ids[index]));
  $('question').textContent=current.question; $('reference').textContent=current.reference_answer;
  $('candidate').textContent=current.candidate_answer;
  $('sources').innerHTML=current.source_records.map(r=>`<div class="panel"><strong>${{esc(r.id)}} — ${{esc(r.title)}}</strong><div class="answer">${{esc(r.statement)}}</div></div>`).join('')||'<p>No source record: OOS/live-source case.</p>';
  const d=current.decision; score=d?.score??null; $('reason').value=d?.reason??'';
  $('confidence').value=d?.confidence??''; $('attention').checked=d?.needs_human_attention??false; $('attest').checked=d?.human_attested??false;
  document.querySelectorAll('.score').forEach(b=>b.classList.toggle('selected',Number(b.dataset.score)===score));
  renderProgress(); $('message').textContent='';
}}
function renderProgress() {{
  $('progressText').textContent=`${{state.completed}} / ${{state.total}} reviewed · case ${{index+1}}`;
  $('progressBar').style.width=`${{100*state.completed/state.total}}%`;
  $('export').disabled=!state.complete;
}}
document.querySelectorAll('.score').forEach(b=>b.onclick=()=>{{score=Number(b.dataset.score);document.querySelectorAll('.score').forEach(x=>x.classList.toggle('selected',x===b));}});
$('save').onclick=async()=>{{
  try {{
    if(score===null)throw new Error('Choose a score.');
    const result=await api('/api/review',{{method:'POST',body:JSON.stringify({{review_id:current.review_id,score,reason:$('reason').value,confidence:$('confidence').value,needs_human_attention:$('attention').checked,human_attested:$('attest').checked}})}});
    state=result.progress; $('message').textContent='Saved.';
    const nextOpen=state.case_ids.findIndex((id,i)=>i>index&&!state.completed_ids.includes(id));
    index=nextOpen>=0?nextOpen:Math.min(index+1,state.total-1); await loadCase();
  }} catch(e){{$('message').textContent=e.message;}}
}};
$('prev').onclick=async()=>{{index=(index-1+state.total)%state.total;await loadCase();}};
$('next').onclick=async()=>{{index=(index+1)%state.total;await loadCase();}};
$('export').onclick=async()=>{{try{{const r=await api('/api/export',{{method:'POST',body:'{{}}'}});$('message').textContent='Exported: '+r.overlay;}}catch(e){{$('message').textContent=e.message;}}}};
$('reviewTab').onclick=()=>{{$('reviewView').classList.remove('hidden');$('statusView').classList.add('hidden');$('reviewTab').classList.add('primary');$('statusTab').classList.remove('primary');}};
$('statusTab').onclick=async()=>{{$('reviewView').classList.add('hidden');$('statusView').classList.remove('hidden');$('statusTab').classList.add('primary');$('reviewTab').classList.remove('primary');const s=await api('/api/status');$('statusGrid').innerHTML=Object.entries(s).map(([k,v])=>`<div class="panel status"><h3>${{esc(k.replaceAll('_',' '))}}</h3><pre>${{esc(typeof v==='string'?v:JSON.stringify(v,null,2))}}</pre></div>`).join('');}};
loadState().catch(e=>{{$('message').textContent=e.message;}});
</script>
</body>
</html>"""

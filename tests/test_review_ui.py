from __future__ import annotations

import hashlib
import http.client
import json
import threading
import zipfile
from pathlib import Path

import pytest

from enterprise_memory_mlx import cli
from enterprise_memory_mlx.review_ui import (
    ReviewDataError,
    ReviewStore,
    create_server,
)


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _packet(tmp_path: Path) -> tuple[Path, Path]:
    cases = [
        {
            "review_id": "RV3-AAA",
            "question": "Question one?",
            "reference_answer": "Reference one.",
            "candidate_answer": "Candidate one.",
            "source_record_ids": ["REC-001"],
        },
        {
            "review_id": "RV3-BBB",
            "question": "Question two?",
            "reference_answer": "Reference two.",
            "candidate_answer": "Candidate two.",
            "source_record_ids": [],
        },
    ]
    sources = [
        {
            "id": "REC-001",
            "domain": "test",
            "title": "Test record",
            "statement": "Authoritative test statement.",
            "source_uri": "synthetic://test/1",
        }
    ]
    members = {
        "REVIEW_INSTRUCTIONS.md": b"Review independently.",
        "review_cases.jsonl": _jsonl(cases),
        "source_records.jsonl": _jsonl(sources),
    }
    manifest = {
        "schema_version": 1,
        "case_count": len(cases),
        "source_candidate": {"cases_sha256": "a" * 64},
        "files": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in members.items()
        },
    }
    packet = tmp_path / "packet.zip"
    with zipfile.ZipFile(packet, "w") as archive:
        for name, content in members.items():
            archive.writestr(f"packet/{name}", content)
        archive.writestr("packet/packet_manifest.json", json.dumps(manifest))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "source_cases_sha256": "a" * 64,
                "mapping": [
                    {"review_id": "RV3-AAA", "case_id": "CASE-001"},
                    {"review_id": "RV3-BBB", "case_id": "CASE-002"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return packet, mapping


def _store(tmp_path: Path) -> ReviewStore:
    packet, mapping = _packet(tmp_path)
    return ReviewStore(
        root=tmp_path,
        packet_path=packet,
        mapping_path=mapping,
        reviewer_id="Human One",
        state_root=tmp_path / "state",
    )


def test_store_exposes_only_blinded_review_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)

    case = store.case("RV3-AAA")

    assert set(case) == {
        "review_id",
        "question",
        "reference_answer",
        "candidate_answer",
        "source_records",
        "decision",
    }
    assert case["source_records"][0]["id"] == "REC-001"
    serialized = json.dumps(case)
    assert "CASE-001" not in serialized
    assert "proposed_semantic_score" not in serialized
    assert store.overview()["completed"] == 0


def test_save_requires_human_attestation_and_case_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ReviewDataError, match="attestation"):
        store.save(
            review_id="RV3-AAA",
            score=1.0,
            reason="Correct.",
            confidence="high",
            needs_human_attention=False,
            human_attested=False,
        )
    with pytest.raises(ReviewDataError, match="reason"):
        store.save(
            review_id="RV3-AAA",
            score=1.0,
            reason="",
            confidence="high",
            needs_human_attention=False,
            human_attested=True,
        )


def test_save_is_atomic_and_resumes_for_same_reviewer(tmp_path: Path) -> None:
    packet, mapping = _packet(tmp_path)
    kwargs = {
        "root": tmp_path,
        "packet_path": packet,
        "mapping_path": mapping,
        "reviewer_id": "Human One",
        "state_root": tmp_path / "state",
    }
    first = ReviewStore(**kwargs)
    first.save(
        review_id="RV3-AAA",
        score=0.5,
        reason="Material timing detail is omitted.",
        confidence="medium",
        needs_human_attention=True,
        human_attested=True,
    )

    resumed = ReviewStore(**kwargs)

    assert resumed.completed == 1
    decision = resumed.case("RV3-AAA")["decision"]
    assert decision["score"] == 0.5
    assert decision["reviewer_kind"] == "human"
    assert decision["needs_human_attention"] is True
    assert decision["identity_verification"] == "asserted_only_not_authenticated"
    assert decision["os_user"]  # invoking OS account is always recorded


def test_export_requires_completion_and_uses_private_mapping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.save(
        review_id="RV3-AAA",
        score=1.0,
        reason="Fully correct.",
        confidence="high",
        needs_human_attention=False,
        human_attested=True,
    )
    with pytest.raises(ReviewDataError, match="incomplete"):
        store.export_overlay()
    store.save(
        review_id="RV3-BBB",
        score=0.0,
        reason="The live fact is unsupported.",
        confidence="high",
        needs_human_attention=False,
        human_attested=True,
    )

    overlay, manifest = store.export_overlay()

    rows = [
        json.loads(line)
        for line in overlay.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["case_id"] for row in rows] == ["CASE-001", "CASE-002"]
    assert all(row["human_reviewers"] == ["Human One"] for row in rows)
    assert all(row["human_approved"] is False for row in rows)
    assert all(
        row["identity_verification"] == "asserted_only_not_authenticated"
        for row in rows
    )
    assert all(row["os_user"] for row in rows)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["status"] == "single_human_review_complete_not_adjudicated"
    assert metadata["identity_verification"] == "asserted_only_not_authenticated"
    assert metadata["os_users"]
    assert metadata["requires_second_review"] is True
    assert metadata["overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()


def test_http_requires_nonce_and_sets_security_headers(tmp_path: Path) -> None:
    packet, mapping = _packet(tmp_path)
    server = create_server(
        root=tmp_path,
        packet_path=packet,
        mapping_path=mapping,
        reviewer_id="Human One",
        port=0,
        state_root=tmp_path / "state",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address)
        connection.request("GET", "/api/state")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Security-Policy")
        state = json.loads(response.read())
        assert state["total"] == 2

        body = json.dumps(
            {
                "review_id": "RV3-AAA",
                "score": 1.0,
                "reason": "Correct.",
                "confidence": "high",
                "human_attested": True,
            }
        )
        connection.request(
            "POST",
            "/api/review",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        assert connection.getresponse().status == 403

        connection.request(
            "POST",
            "/api/review",
            body=body,
            headers={
                "Content-Type": "application/json",
                "X-Review-Nonce": server.nonce,
            },
        )
        saved = connection.getresponse()
        assert saved.status == 200
        assert json.loads(saved.read())["progress"]["completed"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_status_dashboard_is_explicitly_blocked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    packet, mapping = store.packet_path, store.mapping_path
    server = create_server(
        root=tmp_path,
        packet_path=packet,
        mapping_path=mapping,
        reviewer_id="Human One",
        port=0,
        state_root=tmp_path / "other-state",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        status = json.loads(response.read())
        assert status["overall"] == "NOT READY FOR CONFIRMATORY TRAINING"
        assert status["training"] == "confirmatory_blocked_smoke_only"
        assert status["promotion"] == "blocked"
        assert status["benchmark"]["bm25"] == "decision_missing"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cli_review_delegates_to_local_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet, mapping = _packet(tmp_path)
    captured: dict = {}

    def fake_serve(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "serve_review_ui", fake_serve)

    exit_code = cli.main(
        [
            "--root",
            str(tmp_path),
            "review",
            "--packet",
            str(packet),
            "--mapping",
            str(mapping),
            "--reviewer",
            "Human One",
            "--state-root",
            "artifacts/human-reviews/benchmark-smoke",
            "--port",
            "0",
            "--no-browser",
        ]
    )

    assert exit_code == 0
    assert captured["root"] == tmp_path.resolve()
    assert captured["packet_path"] == packet
    assert captured["mapping_path"] == mapping
    assert captured["reviewer_id"] == "Human One"
    assert captured["state_root"] == (
        tmp_path / "artifacts" / "human-reviews" / "benchmark-smoke"
    )
    assert captured["port"] == 0
    assert captured["open_browser"] is False


def test_cli_review_requires_explicit_reviewer_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No default reviewer: identity must be asserted explicitly per session."""
    packet, mapping = _packet(tmp_path)
    monkeypatch.setattr(cli, "serve_review_ui", lambda **kwargs: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--root",
                str(tmp_path),
                "review",
                "--packet",
                str(packet),
                "--mapping",
                str(mapping),
                "--no-browser",
            ]
        )
    assert excinfo.value.code == 2

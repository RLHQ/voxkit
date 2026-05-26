"""Tests for ``voxkit.core.workspace``: layout, manifest, lock, event mirror."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from voxkit.core.workspace import (
    EventMirror,
    Workspace,
    WorkspaceLockError,
    acquire_lock,
    build_artifact_records,
    chunk_paths,
    file_sha256,
    is_artifact_stale,
    open_workspace,
    params_hash,
    read_manifest,
    release_lock,
    write_manifest,
)


# ── open_workspace ──────────────────────────────────────────────────────────


def test_open_workspace_creates_tree(tmp_path: Path) -> None:
    """Root + work/ + work/chunks/ are all created and dataclass paths line up."""
    root = tmp_path / "ws"
    ws = open_workspace(root)

    assert isinstance(ws, Workspace)
    assert ws.root == root
    assert ws.work == root / "work"
    assert ws.chunks == root / "work" / "chunks"
    assert ws.master_wav == root / "work" / "input.16khz.mono.wav"
    assert ws.manifest_path == root / "manifest.json"
    assert ws.raw_json_path == root / "transcript.raw.json"
    assert ws.voxkit_json_path == root / "transcript.voxkit.json"
    assert ws.srt_path == root / "subtitles.srt"
    assert ws.vtt_path == root / "subtitles.vtt"
    assert ws.events_path == root / "events.ndjson"
    assert ws.hallucinations_log == root / "work" / "hallucinations.log"
    assert ws.merge_log == root / "work" / "merge.json"
    assert ws.timeline_log == root / "work" / "timeline_validation.log"

    # Directories exist on disk.
    assert ws.root.is_dir()
    assert ws.work.is_dir()
    assert ws.chunks.is_dir()


def test_open_workspace_accepts_str_path(tmp_path: Path) -> None:
    """``root`` may be passed as ``str`` (CLI commonly hands strings)."""
    ws = open_workspace(str(tmp_path / "ws_str"))
    assert ws.root == tmp_path / "ws_str"
    assert ws.chunks.is_dir()


def test_open_workspace_idempotent(tmp_path: Path) -> None:
    """Calling twice on the same path is a no-op the second time."""
    root = tmp_path / "ws"
    ws1 = open_workspace(root)
    # Plant a file under work/ — second open without force must keep it.
    keepme = ws1.work / "keepme.txt"
    keepme.write_text("hello", encoding="utf-8")

    ws2 = open_workspace(root)
    assert ws1 == ws2
    assert keepme.read_text(encoding="utf-8") == "hello"


def test_open_workspace_force_wipes_work(tmp_path: Path) -> None:
    """``force=True`` removes existing ``work/`` before re-creating."""
    root = tmp_path / "ws"
    ws = open_workspace(root)
    victim = ws.work / "stale.txt"
    victim.write_text("zap me", encoding="utf-8")
    assert victim.exists()

    ws2 = open_workspace(root, force=True)
    assert not victim.exists()
    assert ws2.work.is_dir()
    assert ws2.chunks.is_dir()


# ── chunk_paths ─────────────────────────────────────────────────────────────


def test_chunk_paths_zero(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    wav, full, entries = chunk_paths(ws, 0)
    assert wav == ws.chunks / "chunk_000.wav"
    assert full == ws.chunks / "chunk_000.json"
    assert entries == ws.chunks / "chunk_000.entries.json"


def test_chunk_paths_padding(tmp_path: Path) -> None:
    """Index 42 → ``chunk_042``; index 999 → ``chunk_999``."""
    ws = open_workspace(tmp_path / "ws")
    wav42, full42, entries42 = chunk_paths(ws, 42)
    assert wav42.name == "chunk_042.wav"
    assert full42.name == "chunk_042.json"
    assert entries42.name == "chunk_042.entries.json"

    wav999, _, _ = chunk_paths(ws, 999)
    assert wav999.name == "chunk_999.wav"


# ── manifest I/O ────────────────────────────────────────────────────────────


def test_write_manifest_format(tmp_path: Path) -> None:
    """Manifest is JSON with 2-space indent + trailing newline + UTF-8."""
    ws = open_workspace(tmp_path / "ws")
    write_manifest(ws, {"foo": "bar", "中文": 1})
    text = ws.manifest_path.read_text(encoding="utf-8")
    # Trailing newline.
    assert text.endswith("\n")
    # 2-space indent (look for indented "foo" key).
    assert '\n  "foo"' in text
    # ensure_ascii=False — Chinese stays as glyphs, not \uXXXX escapes.
    assert "中文" in text
    # Round-trips back.
    assert json.loads(text) == {"foo": "bar", "中文": 1}


def test_write_manifest_overwrites(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    write_manifest(ws, {"v": 1})
    write_manifest(ws, {"v": 2, "extra": True})
    assert read_manifest(ws) == {"v": 2, "extra": True}


def test_write_manifest_does_not_leave_tmp(tmp_path: Path) -> None:
    """The atomic ``.json.tmp`` sibling must not survive a successful write."""
    ws = open_workspace(tmp_path / "ws")
    write_manifest(ws, {"v": 1})
    siblings = sorted(p.name for p in ws.root.iterdir() if p.is_file())
    assert "manifest.json" in siblings
    assert "manifest.json.tmp" not in siblings


def test_read_manifest_missing(tmp_path: Path) -> None:
    """Returns ``None`` when the manifest does not yet exist."""
    ws = open_workspace(tmp_path / "ws")
    assert read_manifest(ws) is None


def test_read_manifest_round_trip(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    payload = {"alpha": [1, 2, 3], "beta": {"nested": True}}
    write_manifest(ws, payload)
    assert read_manifest(ws) == payload


def test_file_sha256_and_params_hash_are_stable(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello\n", encoding="utf-8")

    assert file_sha256(p) == file_sha256(p)
    assert len(file_sha256(p)) == 64
    assert params_hash({"b": 2, "a": 1}) == params_hash({"a": 1, "b": 2})
    assert params_hash(None) is None


def test_build_artifact_records_with_sources_and_params(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    raw = ws.raw_json_path
    cues = ws.cues_json_path
    raw.write_text('{"schemaVersion":"1","segments":[]}\n', encoding="utf-8")
    cues.write_text(
        '{"schemaVersion":"1","cues":[{"start":0,"end":1,"text":"hi"}]}\n',
        encoding="utf-8",
    )

    records = build_artifact_records(
        {"raw_json": raw, "subtitle_cues_json": cues, "missing": ws.srt_path},
        root=ws.root,
        created_at="2026-01-01T00:00:00+00:00",
        metadata={
            "subtitle_cues_json": {
                "source_artifacts": ["raw_json"],
                "params": {"resegment": "semantic", "max_chars": 84},
            }
        },
    )

    by_kind = {record["kind"]: record for record in records}
    assert set(by_kind) == {"raw_json", "subtitle_cues_json"}
    assert by_kind["raw_json"]["path"] == "transcript.raw.json"
    assert by_kind["raw_json"]["schemaVersion"] == "1"
    assert len(by_kind["raw_json"]["hash"]) == 64
    cue_record = by_kind["subtitle_cues_json"]
    assert cue_record["sourceArtifacts"] == ["raw_json"]
    assert cue_record["sourceArtifactHashes"] == {
        "raw_json": by_kind["raw_json"]["hash"]
    }
    assert cue_record["paramsHash"] == params_hash(
        {"max_chars": 84, "resegment": "semantic"}
    )
    assert cue_record["status"] == "current"
    assert cue_record["createdAt"] == "2026-01-01T00:00:00+00:00"


def test_is_artifact_stale_detects_source_and_params_changes() -> None:
    record = {
        "paramsHash": params_hash({"model": "a"}),
        "sourceArtifactHashes": {"raw_json": "old"},
    }

    assert not is_artifact_stale(
        record,
        source_hashes={"raw_json": "old"},
        current_params_hash=params_hash({"model": "a"}),
    )
    assert is_artifact_stale(record, source_hashes={"raw_json": "new"})
    assert is_artifact_stale(record, current_params_hash=params_hash({"model": "b"}))


# ── lock ────────────────────────────────────────────────────────────────────


def test_acquire_lock_creates_file(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    acquire_lock(ws)
    lockfile = ws.root / ".lock"
    assert lockfile.is_file()
    body = json.loads(lockfile.read_text(encoding="utf-8"))
    assert body["pid"] == os.getpid()
    assert isinstance(body["started_at"], str) and body["started_at"]


def test_acquire_lock_reentrant_same_process(tmp_path: Path) -> None:
    """Re-acquiring our own lock is a silent no-op."""
    ws = open_workspace(tmp_path / "ws")
    acquire_lock(ws)
    # Capture the exact body — second acquire must not corrupt it.
    first_body = (ws.root / ".lock").read_text(encoding="utf-8")
    acquire_lock(ws)
    second_body = (ws.root / ".lock").read_text(encoding="utf-8")
    assert first_body == second_body


def test_acquire_lock_stale_pid_takes_over(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lock left by a now-dead PID is replaced with a stderr warning."""
    ws = open_workspace(tmp_path / "ws")
    # Plant a fake stale lock — PID 999999 is essentially guaranteed not to exist.
    stale_pid = 999_999
    (ws.root / ".lock").write_text(
        json.dumps({"pid": stale_pid, "started_at": "2020-01-01T00:00:00+00:00"})
        + "\n",
        encoding="utf-8",
    )
    acquire_lock(ws)
    captured = capsys.readouterr()
    assert "stale workspace lock" in captured.err
    assert str(stale_pid) in captured.err
    body = json.loads((ws.root / ".lock").read_text(encoding="utf-8"))
    assert body["pid"] == os.getpid()


def test_acquire_lock_live_foreign_pid_raises(tmp_path: Path) -> None:
    """When another live process owns the lock, raise WorkspaceLockError."""
    ws = open_workspace(tmp_path / "ws")
    # Spawn a real subprocess that sleeps; its PID is guaranteed alive.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # Give it a moment to actually be running.
        time.sleep(0.05)
        (ws.root / ".lock").write_text(
            json.dumps({"pid": proc.pid, "started_at": "2024-01-01T00:00:00+00:00"})
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(WorkspaceLockError) as excinfo:
            acquire_lock(ws)
        assert str(proc.pid) in str(excinfo.value)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_acquire_lock_corrupt_lock_is_replaced(tmp_path: Path) -> None:
    """A garbage ``.lock`` file is overwritten with our PID."""
    ws = open_workspace(tmp_path / "ws")
    (ws.root / ".lock").write_text("not json {{{", encoding="utf-8")
    acquire_lock(ws)
    body = json.loads((ws.root / ".lock").read_text(encoding="utf-8"))
    assert body["pid"] == os.getpid()


def test_release_lock_removes_file(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    acquire_lock(ws)
    assert (ws.root / ".lock").exists()
    release_lock(ws)
    assert not (ws.root / ".lock").exists()


def test_release_lock_missing_is_noop(tmp_path: Path) -> None:
    ws = open_workspace(tmp_path / "ws")
    # No acquire_lock first — must not raise.
    release_lock(ws)
    assert not (ws.root / ".lock").exists()


# ── EventMirror ─────────────────────────────────────────────────────────────


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_event_mirror_emit_method(tmp_path: Path) -> None:
    """``em.emit({...})`` writes one ``\\n``-terminated JSON line."""
    ws = open_workspace(tmp_path / "ws")
    with EventMirror(ws) as em:
        em.emit({"event": "x", "n": 1})
    text = ws.events_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = _read_lines(ws.events_path)
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "x", "n": 1}


def test_event_mirror_callable(tmp_path: Path) -> None:
    """``em({...})`` is equivalent to ``em.emit({...})``."""
    ws = open_workspace(tmp_path / "ws")
    with EventMirror(ws) as em:
        em({"event": "y"})
        em.emit({"event": "z"})
    lines = _read_lines(ws.events_path)
    assert [json.loads(line) for line in lines] == [
        {"event": "y"},
        {"event": "z"},
    ]


def test_event_mirror_appends_across_reentry(tmp_path: Path) -> None:
    """Re-entering the context manager appends — never truncates."""
    ws = open_workspace(tmp_path / "ws")
    with EventMirror(ws) as em:
        em({"event": "a"})
    with EventMirror(ws) as em:
        em({"event": "b"})
        em({"event": "c"})
    lines = _read_lines(ws.events_path)
    assert [json.loads(line) for line in lines] == [
        {"event": "a"},
        {"event": "b"},
        {"event": "c"},
    ]
    # Each line ends with its own terminator: total bytes ≡ sum of (line+\n).
    raw = ws.events_path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 3


def test_event_mirror_unicode(tmp_path: Path) -> None:
    """Non-ASCII payloads stay as glyphs (``ensure_ascii=False``)."""
    ws = open_workspace(tmp_path / "ws")
    with EventMirror(ws) as em:
        em({"event": "speak", "text": "你好喵"})
    lines = _read_lines(ws.events_path)
    assert "你好喵" in lines[0]
    assert json.loads(lines[0]) == {"event": "speak", "text": "你好喵"}

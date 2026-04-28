"""Workspace layout + lock + event-mirror primitives for the transcribe pipeline.

A ``Workspace`` is the single source of truth for every path inside a
``<workdir>/`` directory tree. The layout is *data-orthogonal*: the structure
is defined here, never duplicated across pipeline stages.

Layout::

    <workdir>/
      manifest.json                       # run metadata (no lock entanglement)
      transcript.raw.json                 # whisper.cpp pass-through
      transcript.voxkit.json              # voxkit-native rich transcript
      subtitles.srt
      subtitles.vtt
      events.ndjson                       # mirrored progress events
      .lock                               # PID lock (separate from manifest)
      work/
        input.16khz.mono.wav              # canonical 16kHz mono master
        chunks/
          chunk_NNN.wav
          chunk_NNN.json
          chunk_NNN.entries.json
        hallucinations.log
        merge.json
        timeline_validation.log

Public surface:

* :func:`open_workspace` — create the directory tree, return a frozen
  :class:`Workspace` dataclass.
* :func:`chunk_paths` — derive the three per-chunk paths for index ``idx``.
* :func:`write_manifest` / :func:`read_manifest` — atomic JSON manifest I/O.
* :func:`acquire_lock` / :func:`release_lock` — PID-based co-operative lock
  via a separate ``.lock`` file (no manifest entanglement).
* :class:`EventMirror` — context manager that mirrors NDJSON events to
  ``events.ndjson``.

The lock lives in its own file so :func:`write_manifest` can freely overwrite
``manifest.json`` without round-tripping the lock payload.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = [
    "Workspace",
    "WorkspaceLockError",
    "EventMirror",
    "open_workspace",
    "chunk_paths",
    "write_manifest",
    "read_manifest",
    "acquire_lock",
    "release_lock",
]


# ── Workspace dataclass ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Workspace:
    """All paths inside a transcribe workspace. Frozen — no accidental mutation.

    Every field is an absolute :class:`~pathlib.Path`. Construct via
    :func:`open_workspace` rather than directly so the tree is created on disk.
    """

    root: Path
    work: Path
    chunks: Path
    master_wav: Path
    manifest_path: Path
    raw_json_path: Path
    voxkit_json_path: Path
    srt_path: Path
    vtt_path: Path
    events_path: Path
    hallucinations_log: Path
    merge_log: Path
    timeline_log: Path


def _build_workspace(root: Path) -> Workspace:
    """Pure path construction — no filesystem side-effects."""
    work = root / "work"
    chunks = work / "chunks"
    return Workspace(
        root=root,
        work=work,
        chunks=chunks,
        master_wav=work / "input.16khz.mono.wav",
        manifest_path=root / "manifest.json",
        raw_json_path=root / "transcript.raw.json",
        voxkit_json_path=root / "transcript.voxkit.json",
        srt_path=root / "subtitles.srt",
        vtt_path=root / "subtitles.vtt",
        events_path=root / "events.ndjson",
        hallucinations_log=work / "hallucinations.log",
        merge_log=work / "merge.json",
        timeline_log=work / "timeline_validation.log",
    )


def open_workspace(root: Path | str, *, force: bool = False) -> Workspace:
    """Create the workspace directory tree and return a populated :class:`Workspace`.

    Args:
        root: workspace root directory (created if missing).
        force: when ``True`` and ``<root>/work`` exists, ``rm -rf`` it before
            re-creating. Used by ``--force`` / ``--no-resume`` flows. When
            ``False`` (default), an existing ``work/`` is left in place to
            support resume mode.

    The operation is idempotent for empty / new directories: calling twice on
    the same path with ``force=False`` is a no-op on the second call.
    """
    root_path = Path(root)
    ws = _build_workspace(root_path)

    if force and ws.work.exists():
        shutil.rmtree(ws.work)

    ws.root.mkdir(parents=True, exist_ok=True)
    ws.work.mkdir(parents=True, exist_ok=True)
    ws.chunks.mkdir(parents=True, exist_ok=True)
    return ws


def chunk_paths(ws: Workspace, idx: int) -> tuple[Path, Path, Path]:
    """Return ``(chunk.wav, chunk.json, chunk.entries.json)`` for index ``idx``.

    Index width is 3 zero-padded (``chunk_000`` … ``chunk_999``).
    """
    name = f"chunk_{idx:03d}"
    return (
        ws.chunks / f"{name}.wav",
        ws.chunks / f"{name}.json",
        ws.chunks / f"{name}.entries.json",
    )


# ── Manifest I/O ────────────────────────────────────────────────────────────


def write_manifest(ws: Workspace, manifest: dict[str, Any]) -> None:
    """Atomically write ``manifest`` to ``<root>/manifest.json``.

    Always overwrites. Format: ``json.dumps(..., ensure_ascii=False, indent=2)``
    plus a trailing newline. Atomicity is guaranteed by writing to a sibling
    ``.json.tmp`` file and ``os.replace``-ing it into place.
    """
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    tmp = ws.manifest_path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, ws.manifest_path)


def read_manifest(ws: Workspace) -> dict[str, Any] | None:
    """Return the parsed manifest dict, or ``None`` when ``manifest.json`` is absent."""
    try:
        text = ws.manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return json.loads(text)


# ── PID lock ────────────────────────────────────────────────────────────────


class WorkspaceLockError(RuntimeError):
    """Raised when another *live* voxkit process owns the workspace lock."""


def _lock_payload() -> tuple[str, dict[str, Any]]:
    pid = os.getpid()
    started = datetime.now(timezone.utc).isoformat()
    payload = {"pid": pid, "started_at": started}
    return json.dumps(payload) + "\n", payload


def _is_pid_alive(pid: int) -> bool:
    """Return whether ``pid`` is currently alive on this OS.

    Uses ``os.kill(pid, 0)``: ``ESRCH`` ⇒ dead; ``EPERM`` ⇒ alive but not ours;
    anything else (success or other errno) ⇒ treat as alive.
    """
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        # Unexpected errno — be conservative and assume alive.
        return True
    return True


def acquire_lock(ws: Workspace) -> None:
    """Take the workspace lock by writing this process's PID to ``<root>/.lock``.

    Semantics:

    * If no lock exists → atomically create one (``O_CREAT|O_EXCL``).
    * If lock exists and points to *us* (same PID) → no-op (re-entry).
    * If lock exists and points to a *dead* PID → take over with a stderr warning.
    * If lock exists and points to a *live foreign* PID → raise
      :class:`WorkspaceLockError` (message includes the live PID).
    * If lock file is unreadable / corrupt → replace it.

    Liveness is checked via :func:`os.kill` with signal 0.
    """
    lockfile = ws.root / ".lock"
    text, _payload = _lock_payload()

    # Fast path: atomic create.
    try:
        fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, text.encode("utf-8"))
        finally:
            os.close(fd)
        return
    except FileExistsError:
        pass

    # Lock already exists — inspect it.
    try:
        existing = json.loads(lockfile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Corrupt or unreadable — replace.
        lockfile.write_text(text, encoding="utf-8")
        return

    other_pid_raw = existing.get("pid") if isinstance(existing, dict) else None
    if other_pid_raw is None:
        lockfile.write_text(text, encoding="utf-8")
        return

    try:
        other_pid = int(other_pid_raw)
    except (TypeError, ValueError):
        lockfile.write_text(text, encoding="utf-8")
        return

    # Re-entry: this process already owns the lock.
    if other_pid == os.getpid():
        return

    if _is_pid_alive(other_pid):
        raise WorkspaceLockError(
            f"workspace {ws.root} is locked by live PID {other_pid} "
            f"(started {existing.get('started_at')})"
        )

    # Stale lock — take over.
    sys.stderr.write(
        f"warning: stale workspace lock for dead PID {other_pid}; taking over\n"
    )
    lockfile.write_text(text, encoding="utf-8")


def release_lock(ws: Workspace) -> None:
    """Remove ``<root>/.lock`` if present. Best-effort, no-op when missing."""
    try:
        (ws.root / ".lock").unlink()
    except FileNotFoundError:
        pass


# ── Event mirror ────────────────────────────────────────────────────────────


class EventMirror:
    """Context manager that mirrors NDJSON events to ``<root>/events.ndjson``.

    Usage::

        with EventMirror(ws) as emit:
            emit({"event": "progress", "stage": "whisper.chunk", "percent": 25})

    The context value supports both ``emit.emit({...})`` and ``emit({...})``
    (callable form). Each call writes one JSON line, ``\\n``-terminated,
    UTF-8 encoded, in append mode — the file grows monotonically across
    multiple ``with``-block re-entries.

    Forwarding events to stderr (e.g. for ``--json-events``) is a separate
    concern; this class only owns the file mirror.
    """

    def __init__(self, ws: Workspace) -> None:
        self._path: Path = ws.events_path
        self._fp: Any = None

    def __enter__(self) -> "EventMirror":
        # Ensure parent dir exists (defensive: open_workspace already created it).
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._path, "a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        fp = self._fp
        self._fp = None
        if fp is not None:
            try:
                fp.flush()
            finally:
                fp.close()

    def emit(self, event: dict[str, Any]) -> None:
        """Write one JSON line to ``events.ndjson``."""
        line = json.dumps(event, ensure_ascii=False) + "\n"
        if self._fp is not None:
            self._fp.write(line)
            self._fp.flush()
            return
        # Fallback: not inside a ``with`` block — open/append/close per call so
        # ``EventMirror(ws).emit({...})`` still works as a one-shot.
        with open(self._path, "a", encoding="utf-8") as fp:
            fp.write(line)

    def __call__(self, event: dict[str, Any]) -> None:
        self.emit(event)

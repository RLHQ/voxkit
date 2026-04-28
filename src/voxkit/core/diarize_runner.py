"""pyannote diarization worker spawn — extracted from ``commands/diarize.py``.

Responsibility split:

* ``commands/diarize.py`` — CLI handler: parse args, prepare audio, write
  artefacts, optionally call into ``commands/align.py`` for the SRT output.
* ``core/diarize_runner.py`` (this module) — pure runner: build argv, spawn
  the worker process inside the lazy-install venv, parse the sentinel-prefixed
  JSON line on stdout, raise typed exceptions on failure.

The runner is callable from BOTH the diarize CLI and the transcribe pipeline
(when ``--with-diarization`` is on). Both surfaces share argv / sentinel /
exit-code semantics through this module.

The worker entry point is ``voxkit.core.pipeline:main`` (see ``pipeline.py``).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from voxkit.core.constants import WORKER_JSON_SENTINEL
from voxkit.io.progress import ProgressEmitter
from voxkit.io.schema import DiarizationOutput

__all__ = [
    "run_diarize",
    "DiarizeFailed",
    "DiarizeTimeout",
    "build_worker_argv",
    "extract_sentinel_json",
]


# Stderr progress regex — matches the human-readable form the worker's
# ``ProgressEmitter`` emits when ``--json-events`` is OFF, e.g.
# ``[diarize] 42%``. We do not parse the NDJSON form here because the runner
# spawns the worker without ``--json-events`` by default (matches the existing
# ``commands/diarize.py`` behaviour: human stderr is forwarded verbatim to the
# parent terminal).
_STDERR_PROGRESS_RE = re.compile(r"\[(?P<stage>[a-zA-Z_][a-zA-Z0-9_]*)\]\s+(?P<pct>\d+)%")


class DiarizeFailed(RuntimeError):
    """The pyannote worker exited with a non-zero return code.

    Attributes:
        returncode: subprocess return code.
        stderr_tail: tail of stderr (truncated to the last ~50 lines) for
            error reporting.
    """

    def __init__(self, returncode: int, stderr_tail: str):
        super().__init__(
            f"voxkit diarize worker failed with returncode={returncode}\n"
            f"--- stderr tail ---\n{stderr_tail}"
        )
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class DiarizeTimeout(RuntimeError):
    """The worker did not finish within ``timeout_secs``."""


def build_worker_argv(
    *,
    venv_python: Path,
    audio_path: Path,
    duration_secs: float,
    speaker_labels: str = "ranked",
    model: str = "sd-3.1",
    device: str = "auto",
    extracted_from: Path | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    json_events: bool = False,
) -> list[str]:
    """Build argv for ``<venv_python> -m voxkit.core.pipeline ...``.

    Pure function — extracted for unit-testability.
    """
    argv: list[str] = [
        str(venv_python),
        "-m", "voxkit.core.pipeline",
        "--audio", str(audio_path),
        "--audio-duration-secs", f"{duration_secs:.6f}",
        "--model", model,
        "--device", device,
        "--speaker-labels", speaker_labels,
    ]
    if extracted_from is not None:
        argv += ["--extracted-from", str(extracted_from)]
    if num_speakers is not None:
        argv += ["--num-speakers", str(num_speakers)]
    if min_speakers is not None:
        argv += ["--min-speakers", str(min_speakers)]
    if max_speakers is not None:
        argv += ["--max-speakers", str(max_speakers)]
    if json_events:
        argv += ["--json-events"]
    return argv


def extract_sentinel_json(stdout_text: str) -> Optional[str]:
    """Find the first ``__VOXKIT_JSON__``-prefixed line in ``stdout_text`` and
    return the JSON payload (sentinel stripped). Returns ``None`` if no
    sentinel line is present.
    """
    for ln in stdout_text.splitlines():
        if ln.startswith(WORKER_JSON_SENTINEL):
            return ln[len(WORKER_JSON_SENTINEL):]
    return None


def _emit_progress_lines(
    stderr_text: str, progress: ProgressEmitter | None
) -> None:
    """Parse ``[stage] N%`` lines from stderr and forward to ``progress``.

    Best-effort — exceptions in the callback are swallowed to never break a
    successful run on a misbehaving emitter.
    """
    if progress is None:
        return
    for line in stderr_text.splitlines():
        m = _STDERR_PROGRESS_RE.search(line)
        if not m:
            continue
        try:
            pct = int(m.group("pct"))
        except ValueError:
            continue
        try:
            progress.progress(m.group("stage"), pct)
        except Exception:  # noqa: BLE001 — never let the emitter break the pipeline
            pass


def run_diarize(
    audio_path: Path,
    *,
    duration_secs: float,
    venv_python: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model: str = "sd-3.1",
    device: str = "auto",
    speaker_labels: str = "ranked",
    extracted_from: Path | None = None,
    progress: ProgressEmitter | None = None,
    env: dict[str, str] | None = None,
    timeout_secs: float | None = None,
    forward_stderr: bool = True,
    json_events: bool = False,
) -> DiarizationOutput:
    """Spawn the pyannote worker subprocess and return parsed
    :class:`DiarizationOutput`.

    Args:
        audio_path: 16kHz mono wav path (the worker is happiest with this; it
            re-decodes anything else via torchaudio but adds latency).
        duration_secs: Probed duration; fed to the worker for RTF calculation.
        venv_python: ``<venv>/bin/python`` from
            :func:`voxkit.core.lazy_install.ensure_venv`.
        num_speakers / min_speakers / max_speakers: Optional pyannote
            constraints. Passed through verbatim.
        model: short alias from ``MODEL_ALIASES`` (``"sd-3.1"`` /
            ``"community-1"``). Defaults to ``"sd-3.1"``.
        device: ``"auto"`` / ``"mps"`` / ``"cuda"`` / ``"cpu"``.
        speaker_labels: ``"ranked"`` (default) or ``"raw"``. Used purely for
            the worker's own ``DiarizationOutput.speakers[*].id`` mapping;
            downstream alignment can choose either independently.
        extracted_from: Original video path when ``audio_path`` was extracted
            from one. Recorded into the output's ``audio.extractedFrom``.
        progress: Optional progress emitter; we scan the captured stderr for
            ``[stage] N%`` lines (the worker's human format) and forward them.
        env: Subprocess environment (typically ``core_env.patched_env()`` so
            the macOS DYLD_LIBRARY_PATH dance applies).
        timeout_secs: Wall-clock cap. ``None`` → no timeout.
        forward_stderr: If True (default), copy worker stderr to ``sys.stderr``
            so the user sees pyannote/torch warnings + progress in real-ish
            time. Set False for headless callers that want pure isolation.
        json_events: Pass ``--json-events`` to the worker. When True the
            worker's stderr is NDJSON; ``forward_stderr`` then yields a clean
            event stream consumable by Remixr-style log eaters. The runner
            still scans for ``[stage] N%`` lines — those simply will not exist
            in NDJSON mode, so the progress callback won't fire from stderr
            (callers can hook the events file mirror instead).

    Returns:
        Parsed :class:`DiarizationOutput`.

    Raises:
        DiarizeFailed: non-zero exit code.
        DiarizeTimeout: ``timeout_secs`` exceeded.
        ValueError: stdout has no sentinel line, or sentinel JSON does not
            validate as :class:`DiarizationOutput`.
    """
    argv = build_worker_argv(
        venv_python=venv_python,
        audio_path=audio_path,
        duration_secs=duration_secs,
        speaker_labels=speaker_labels,
        model=model,
        device=device,
        extracted_from=extracted_from,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        json_events=json_events,
    )

    started = time.monotonic()

    # We capture both streams up-front: simpler than threaded streaming and
    # keeps the existing ``commands/diarize.py`` semantics intact (the original
    # ``_run_worker`` used ``subprocess.run(capture_output=True)``).
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        # subprocess module already killed the child; surface a typed error.
        raise DiarizeTimeout(
            f"voxkit diarize worker exceeded {timeout_secs}s "
            f"(actual {elapsed:.1f}s before timeout)"
        ) from exc

    stderr_text: str = proc.stderr or ""
    stdout_text: str = proc.stdout or ""

    # Forward stderr to the parent terminal. We do this BEFORE checking the
    # return code so the user sees worker diagnostics regardless of outcome.
    if forward_stderr and stderr_text:
        try:
            sys.stderr.write(stderr_text)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 — never block on stderr
            pass

    # Forward progress to the optional emitter. Only meaningful in human-mode
    # stderr (json_events=False); the regex won't match NDJSON.
    _emit_progress_lines(stderr_text, progress)

    if proc.returncode != 0:
        # Best-effort: surface the last ~50 lines of stderr alongside the rc.
        tail_lines = stderr_text.splitlines()[-50:]
        raise DiarizeFailed(
            returncode=proc.returncode,
            stderr_tail="\n".join(tail_lines),
        )

    json_payload = extract_sentinel_json(stdout_text)
    if json_payload is None:
        raise ValueError(
            "voxkit diarize worker stdout has no "
            f"{WORKER_JSON_SENTINEL!r} sentinel line; "
            f"stdout head={stdout_text[:200]!r}"
        )

    try:
        return DiarizationOutput.model_validate_json(json_payload)
    except Exception as exc:
        raise ValueError(
            f"voxkit diarize worker produced invalid sentinel JSON: {exc}\n"
            f"line={json_payload[:200]!r}"
        ) from exc

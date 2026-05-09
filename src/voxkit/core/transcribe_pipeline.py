"""End-to-end transcribe pipeline orchestrator.

Wires together every Round-1 module:

* :mod:`voxkit.core.workspace` — paths + lock + NDJSON event mirror
* :mod:`voxkit.core.audio` — ffmpeg normalize / chunk plan / chunk extract
* :mod:`voxkit.core.whisper_exec` — whisper-cli discovery + spawn + JSON parse
* :mod:`voxkit.core.segmenter` — entry → ``TranscriptSegment`` reshape
* :mod:`voxkit.core.hallucination_filter` — channel watermark / standalone /
  ghost-loop drop
* :mod:`voxkit.core.asr_merge` — chunk-relative → absolute timeline + dedup
* :mod:`voxkit.io.schema` — Pydantic transcript types
* :mod:`voxkit.io.remixr_adapter` — voxkit-native → Remixr-shaped on-disk
* :mod:`voxkit.io.srt` — SRT / VTT renderers

Entry point: :func:`run_pipeline`. Caller is :mod:`voxkit.commands.transcribe`.

Design contracts (do not break without updating callers):

* Lock acquisition wraps the whole run; ``release_lock`` is in ``finally``.
* Event mirroring is unconditional; ``--json-events`` only adds the stderr
  forward, the file mirror always exists for post-mortem.
* ``transcript.raw.json`` is exclusive-write (``mode="x"``). When ``resume`` is
  on and the file already exists, abort with a helpful message; when ``resume``
  is off (i.e. ``--force`` / ``--no-resume``), unlink first to honour force
  semantics.
* Per-chunk checkpoint hits skip both ffmpeg chunk extraction and the
  whisper-cli spawn.
* ``ChunkResult.segments`` is *chunk-relative* — :func:`merge_chunks` does the
  global offset. Do not pre-offset segments here.

The module is import-clean (no side-effects beyond the function bodies)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from voxkit.core.semantic_resegment import SubtitleCue

ResegmentMode = Literal["none", "semantic"]

from voxkit import __version__
from voxkit.core import env as core_env
from voxkit.core.audio import (
    ChunkPlan,
    ChunkSpec,
    chunk_thresholds_from_env,
    extract_chunk,
    find_ffmpeg,
    normalize_to_wav_16k_mono,
    plan_chunks,
    probe_duration,
)
from voxkit.core.asr_merge import (
    ChunkResult,
    MergeNote,
    merge_chunks,
    write_merge_log,
)
from voxkit.core.constants import DIA_PHANTOM_FILTER_S, ExitCode
from voxkit.core.hallucination_filter import (
    Blocklist,
    DroppedEntry,
    filter_entries,
    load_blocklist,
    write_drop_log,
)
from voxkit.core.segmenter import detect_mode, segment_entries
from voxkit.core.types import Entry
from voxkit.core.whisper_exec import (
    CJK_LANGUAGES,
    WhisperFailed,
    WhisperFlags,
    WhisperRunResult,
    WhisperTimeout,
    find_vad_model,
    find_whisper_cli,
    find_whisper_model,
    parse_whisper_json,
    run_whisper,
)
from voxkit.core.workspace import (
    EventMirror,
    Workspace,
    acquire_lock,
    chunk_paths,
    release_lock,
    write_manifest,
)
from voxkit.io.remixr_adapter import to_remixr_transcript, write_remixr_json
from voxkit.io.schema import (
    AudioInfo,
    ChunkStat,
    DiarizationOutput,
    RemixrTranscript,
    TranscriptionOutput,
    TranscriptSegment,
)
from voxkit.io.srt import (
    to_subtitles_srt,
    to_subtitles_vtt,
)

__all__ = [
    "PipelineError",
    "ResegmentMode",
    "TranscribeRequest",
    "TranscribeResult",
    "run_pipeline",
]


# ── Public types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TranscribeRequest:
    """Resolved, validated parameters for a single transcribe run.

    Construct from the CLI ``argparse.Namespace`` in
    :mod:`voxkit.commands.transcribe`. Order of fields matches the
    ``add_subparser`` flag order for review-readability.

    ``with_diarization`` / ``speaker_labels`` (Phase 2) are appended at the
    end with sensible defaults so existing callers that omit them keep their
    v0.3.0 behaviour byte-identical.
    """

    input_path: Path
    workspace: Workspace
    model: str
    language: str
    word_timestamps: bool
    vad: bool
    logprob_thold: float
    source_id: str
    keep_work: bool
    json_events: bool
    timeout_ms: int | None
    whisper_bin_override: Path | None
    vad_model_override: Path | None
    blocklist_path: Path | None
    resume: bool
    emit_srt: bool
    emit_vtt: bool
    # ── Phase 2 — diarization integration ────────────────────────────
    with_diarization: bool = False
    speaker_labels: str = "ranked"
    # ── Phase 3 — semantic subtitle resegmentation ───────────────────
    # "semantic" runs voxkit.core.semantic_resegment as a post-processor
    # (pysbd sentence boundaries + clause-aware splitting; CJK passthrough).
    # Only affects SRT/VTT — JSON outputs are byte-identical regardless.
    resegment: ResegmentMode = "none"


@dataclass(frozen=True)
class TranscribeResult:
    """Successful run summary (returned by :func:`run_pipeline`)."""

    voxkit_output: TranscriptionOutput
    artifacts: dict[str, Path]
    warnings: list[str]
    elapsed_secs: float
    rtf: float


class PipelineError(RuntimeError):
    """Pipeline failed with a user-facing message + exit code.

    Caller (``commands/transcribe.py``) catches this, prints the message to
    stderr, and returns ``exit_code`` from ``main()``.
    """

    def __init__(self, message: str, exit_code: int = int(ExitCode.GENERIC_FAIL)):
        super().__init__(message)
        self.exit_code = exit_code


# ── Video extension set used to populate AudioInfo.extracted_from ──────────
_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".avi"}
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _emit_event(
    em: EventMirror,
    payload: dict[str, Any],
    *,
    forward_to_stderr: bool,
) -> None:
    """Write one event to the file mirror; optionally also to stderr."""
    em.emit(payload)
    if forward_to_stderr:
        try:
            sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 — never let stderr block the run
            pass


def _discover_binaries(
    req: TranscribeRequest,
) -> tuple[Path, Path, Path | None]:
    """Resolve whisper-cli, model, and (optional) VAD model. Fail-fast on first
    two; tolerate missing VAD."""
    whisper_bin = find_whisper_cli(override=req.whisper_bin_override)
    if whisper_bin is None:
        raise PipelineError(
            "whisper-cli not found. Install via `brew install whisper-cpp`, "
            "set $WHISPER_BIN, or pass --whisper-bin <path>.",
            exit_code=int(ExitCode.ENV_PROBLEM),
        )

    model_path = find_whisper_model(req.model)
    if model_path is None:
        raise PipelineError(
            f"whisper model not found for alias/path: {req.model!r}. "
            f"Try `huggingface-cli download ggerganov/whisper.cpp ggml-{req.model}.bin "
            f"--local-dir ~/.cache/voxkit/models`.",
            exit_code=int(ExitCode.ENV_PROBLEM),
        )

    vad_model = None
    if req.vad:
        vad_model = find_vad_model(override=req.vad_model_override)
        # vad=True + missing model is non-fatal — caller adds a warning.

    return whisper_bin, model_path, vad_model


def _normalize_detected_language(value: Any) -> str | None:
    """Return a usable ISO-ish language code from whisper metadata."""
    if not isinstance(value, str):
        return None
    lang = value.strip().lower()
    if not lang or lang in {"auto", "unknown"}:
        return None
    return lang


def _extract_detected_language(raw: dict) -> str | None:
    """Read whisper.cpp's detected language from an ``-ojf`` JSON payload."""
    result = raw.get("result")
    if isinstance(result, dict):
        lang = _normalize_detected_language(result.get("language"))
        if lang:
            return lang

    # Older / synthetic fixtures may only expose params. Treat it as a
    # best-effort fallback; "auto" is filtered by _normalize_detected_language.
    params = raw.get("params")
    if isinstance(params, dict):
        return _normalize_detected_language(params.get("language"))
    return None


def _resolve_auto_language(
    req_language: str,
    detected_languages: list[str],
    inferred_languages: list[str],
    warnings: list[str],
) -> str:
    """Pick the downstream language for output + subtitle resegmentation.

    ``--language auto`` is valid for whisper, but downstream sentence boundary
    tools require a concrete language code. Prefer whisper.cpp's own detection;
    fall back to the same word/phrase heuristic used by the segmenter.
    """
    if req_language != "auto":
        return req_language

    candidates = detected_languages or inferred_languages
    if not candidates:
        warnings.append("language=auto, no language evidence available")
        return req_language

    counts = Counter(candidates)
    language, _ = counts.most_common(1)[0]
    if len(counts) > 1:
        detail = ", ".join(f"{lang}={count}" for lang, count in sorted(counts.items()))
        warnings.append(
            f"language=auto, multiple detected languages ({detail}); using {language}"
        )
    else:
        warnings.append(f"language=auto, resolved language={language}")
    return language


def _prepare_master_wav(
    req: TranscribeRequest,
    em: EventMirror,
    *,
    forward_stderr: bool,
) -> tuple[Path, float]:
    """ffmpeg → 16kHz mono master wav. Skip when checkpoint exists and resume."""
    ws = req.workspace
    master = ws.master_wav

    needs_normalize = not (req.resume and master.exists() and master.stat().st_size > 0)

    if needs_normalize:
        _emit_event(
            em,
            {"event": "audio.normalize.start", "input": str(req.input_path)},
            forward_to_stderr=forward_stderr,
        )
        if find_ffmpeg() is None:
            raise PipelineError(
                "ffmpeg not found. Install via `brew install ffmpeg-full`.",
                exit_code=int(ExitCode.ENV_PROBLEM),
            )
        try:
            normalize_to_wav_16k_mono(req.input_path, master)
        except RuntimeError as exc:
            raise PipelineError(
                f"audio normalize failed: {exc}",
                exit_code=int(ExitCode.GENERIC_FAIL),
            ) from exc

    try:
        duration = probe_duration(master)
    except Exception as exc:  # noqa: BLE001 — wrap into pipeline error
        raise PipelineError(
            f"ffprobe failed on {master}: {exc}",
            exit_code=int(ExitCode.GENERIC_FAIL),
        ) from exc

    _emit_event(
        em,
        {
            "event": "audio.normalize.done",
            "master_wav": str(master),
            "duration_secs": duration,
        },
        forward_to_stderr=forward_stderr,
    )
    return master, duration


def _resolve_chunk_timeout_ms(
    req: TranscribeRequest, spec: ChunkSpec
) -> int:
    """Per-chunk timeout: max(30 min, duration * 0.3); CLI override wins."""
    if req.timeout_ms is not None:
        return int(req.timeout_ms)
    dynamic = max(30 * 60 * 1000, int(spec.duration_secs * 0.3 * 1000))
    return dynamic


def _transcribe_chunk(
    req: TranscribeRequest,
    spec: ChunkSpec,
    *,
    whisper_bin: Path,
    model_path: Path,
    vad_model: Path | None,
    blocklist: Blocklist,
    em: EventMirror,
    forward_stderr: bool,
) -> tuple[list[Entry], list[DroppedEntry], float, bool, str | None]:
    """Run (or resume) one chunk.

    Returns ``(kept_entries, dropped, elapsed, cached, detected_language)``.
    """
    ws = req.workspace
    chunk_wav, chunk_json, _entries_json = chunk_paths(ws, spec.index)

    # 1. Resume detection — checkpoint hit
    cached = False
    raw_dict: dict | None = None
    elapsed = 0.0
    if req.resume and chunk_json.exists() and chunk_json.stat().st_size > 0:
        try:
            # errors="replace": rationale in whisper_exec.run_whisper.
            # Strict decode would force a cache miss on every CJK resume.
            with chunk_json.open("r", encoding="utf-8", errors="replace") as f:
                raw_dict = json.load(f)
            cached = True
        except (OSError, json.JSONDecodeError):
            raw_dict = None
            cached = False

    # 2. Cache miss → ensure chunk wav, then spawn whisper-cli
    if not cached:
        if not chunk_wav.exists() or chunk_wav.stat().st_size == 0:
            try:
                extract_chunk(ws.master_wav, spec)
            except RuntimeError as exc:
                raise PipelineError(
                    f"chunk {spec.index} extract failed: {exc}",
                    exit_code=int(ExitCode.GENERIC_FAIL),
                ) from exc

        flags = WhisperFlags(
            model_path=model_path,
            language=req.language,
            vad=req.vad and vad_model is not None,
            vad_model_path=vad_model,
            logprob_thold=req.logprob_thold,
            word_timestamps=req.word_timestamps,
            max_context_zero=True,
        )

        timeout_secs = _resolve_chunk_timeout_ms(req, spec) / 1000.0

        def _on_progress(pct: int, *, _idx: int = spec.index) -> None:
            _emit_event(
                em,
                {
                    "event": "progress",
                    "stage": "whisper.chunk",
                    "chunk": _idx,
                    "percent": pct,
                },
                forward_to_stderr=forward_stderr,
            )

        try:
            result: WhisperRunResult = run_whisper(
                chunk_wav,
                chunk_json,
                flags,
                whisper_bin=whisper_bin,
                timeout_secs=timeout_secs,
                env=core_env.patched_env(),
                progress_cb=_on_progress,
            )
        except WhisperTimeout as exc:
            raise PipelineError(
                f"chunk {spec.index} whisper-cli timed out after {timeout_secs:.0f}s",
                exit_code=int(ExitCode.GENERIC_FAIL),
            ) from exc
        except WhisperFailed as exc:
            raise PipelineError(
                f"chunk {spec.index} whisper-cli failed (rc={exc.returncode}):\n"
                f"{exc.stderr_tail}",
                exit_code=int(ExitCode.GENERIC_FAIL),
            ) from exc
        raw_dict = result.raw_json
        elapsed = result.elapsed_secs

    assert raw_dict is not None
    detected_language = _extract_detected_language(raw_dict)
    entries = parse_whisper_json(raw_dict)

    # 3. Hallucination filter — entry-level, BEFORE segmentation.
    # Blocklist is loaded once by the caller and reused across chunks.
    kept, dropped = filter_entries(entries, blocklist, chunk_index=spec.index)

    # 4. Persist drops to log (append-mode; resume-friendly)
    if dropped:
        try:
            write_drop_log(dropped, ws.hallucinations_log)
        except OSError:
            # log write failure is non-fatal — caller still has the in-memory list
            pass

    return kept, dropped, elapsed, cached, detected_language


def _build_audio_info(req: TranscribeRequest, duration_secs: float) -> AudioInfo:
    """Construct ``AudioInfo`` for the final transcript.

    ``extracted_from`` is set when the input is a video container; otherwise
    None (audio-native input).
    """
    is_video = req.input_path.suffix.lower() in _VIDEO_EXTS
    return AudioInfo(
        path=str(req.workspace.master_wav),
        durationSecs=float(duration_secs),
        extractedFrom=str(req.input_path) if is_video else None,
    )


def _ensure_raw_json_writable(req: TranscribeRequest) -> None:
    """Honour the ``transcript.raw.json`` exclusive-write contract (Plan §user-decision #1).

    Semantics:

    * Absent → no-op (the common case)
    * Present + ``resume=True`` → ``PipelineError`` (workspace already has a
      completed transcript; user must pick a fresh ``--workdir`` or pass
      ``--force``).
    * Present + ``resume=False`` (i.e. ``--force`` / ``--no-resume``) → unlink
      so the subsequent ``write_remixr_json('x')`` can proceed cleanly.

    Also clears ``subtitles.cues.json`` on ``--force`` for symmetry — that file
    uses the same exclusive-create contract and would otherwise survive a
    ``--force`` rerun and trigger a late-stage ``FileExistsError``.

    This matches the brainstorming-phase decision: ``transcript.raw.json``
    is treated like Remixr's own raw artifacts (write-once-never-overwrite)
    by default; ``--force`` is the explicit opt-out.
    """
    raw_path = req.workspace.raw_json_path
    cues_path = req.workspace.cues_json_path
    if not raw_path.exists():
        # cues_json is only written when raw_json is also fresh — but on a
        # --force rerun that wiped raw_json by hand, we still want to clean it.
        if not req.resume and cues_path.exists():
            cues_path.unlink()
        return
    if req.resume:
        raise PipelineError(
            f"{raw_path} already exists for this workspace; "
            f"transcribe already completed here. "
            f"Pass --force to overwrite, or pick a fresh --workdir.",
            exit_code=ExitCode.GENERIC_FAIL.value,
        )
    raw_path.unlink()
    if cues_path.exists():
        cues_path.unlink()


# ── Phase 2: diarization integration helpers ──────────────────────────────


def _run_diarization_pass(
    req: TranscribeRequest,
    *,
    master_wav: Path,
    duration_secs: float,
    em: EventMirror,
    forward_stderr: bool,
) -> tuple[DiarizationOutput, float]:
    """Run pyannote diarization on the master wav. Returns
    ``(DiarizationOutput, elapsed_secs)``.

    The work is done by ``voxkit.core.diarize_runner.run_diarize`` which spawns
    the pyannote worker inside the lazy-install venv. We always lazy-trigger
    venv creation here; first run in a fresh environment will block on the
    install (1-3 min) — caller decides whether that is acceptable.

    Default model = ``"sd-3.1"`` and device = ``"auto"``. There is no CLI knob
    for these in transcribe today; if needed we'd add ``--diarize-model`` /
    ``--diarize-device`` later.
    """
    # Late imports keep ``transcribe_pipeline`` importable on machines without
    # pyannote installed (the worker is a separate venv anyway).
    from voxkit.core.diarize_runner import (
        DiarizeFailed,
        DiarizeTimeout,
        run_diarize,
    )
    from voxkit.core.lazy_install import SetupError, ensure_venv

    diarize_model = "sd-3.1"
    diarize_device = "auto"

    _emit_event(
        em,
        {
            "event": "diarize.start",
            "model": diarize_model,
            "device": diarize_device,
        },
        forward_to_stderr=forward_stderr,
    )

    # 1. Ensure the lazy venv exists (pyannote + torch).
    try:
        venv_info = ensure_venv(verbose=not req.json_events)
    except SetupError as exc:
        raise PipelineError(
            f"diarization venv setup failed: {exc}",
            exit_code=int(ExitCode.GENERIC_FAIL),
        ) from exc

    # 2. Spawn the worker.
    started_d = time.monotonic()
    try:
        diarization = run_diarize(
            master_wav,
            duration_secs=duration_secs,
            venv_python=venv_info.venv_python,
            model=diarize_model,
            device=diarize_device,
            speaker_labels=req.speaker_labels,
            extracted_from=req.input_path
            if req.input_path.suffix.lower() in _VIDEO_EXTS
            else None,
            env=core_env.patched_env(),
            forward_stderr=True,
            json_events=req.json_events,
        )
    except DiarizeFailed as exc:
        raise PipelineError(
            f"diarization worker failed (rc={exc.returncode}):\n{exc.stderr_tail}",
            exit_code=int(ExitCode.WORKER_FAILED),
        ) from exc
    except DiarizeTimeout as exc:
        raise PipelineError(
            f"diarization worker timed out: {exc}",
            exit_code=int(ExitCode.WORKER_FAILED),
        ) from exc
    except ValueError as exc:
        # sentinel missing or invalid JSON
        raise PipelineError(
            f"diarization worker produced invalid output: {exc}",
            exit_code=int(ExitCode.WORKER_FAILED),
        ) from exc

    elapsed_d = time.monotonic() - started_d
    return diarization, elapsed_d


def _remixr_to_cues(t: RemixrTranscript) -> "list[SubtitleCue]":
    """Adapt a diarized RemixrTranscript to ``SubtitleCue[]`` for the cue
    renderer. One cue per segment; speaker label survives. Used both when
    resegment is off (legacy 1-cue-per-segment) and as a typed bridge so the
    SRT / VTT renderers have a single code path.
    """
    from voxkit.core.semantic_resegment import SubtitleCue
    return [
        SubtitleCue(start=s.start, end=s.end, speaker=s.speaker, text=s.text.strip())
        for s in t.segments
    ]


# ── Main entry point ──────────────────────────────────────────────────────


def run_pipeline(req: TranscribeRequest) -> TranscribeResult:
    """End-to-end transcribe orchestration.

    Steps:

      1. Acquire workspace lock; honour ``transcript.raw.json`` write contract.
      2. Discover binaries (whisper-cli, model, optional VAD).
      3. ffmpeg → 16kHz master wav (skip on resume + cached file).
      4. ``plan_chunks`` from probed duration.
      5. Per-chunk: resume / extract / whisper / parse / filter.
      6. Per-chunk: segment_entries on chunk-relative entries.
      7. ``merge_chunks`` → absolute timeline + dedup.
      8. Build :class:`TranscriptionOutput`.
      9. Write ``transcript.voxkit.json`` (camelCase, ``by_alias=True``).
     10. Map → Remixr; write ``transcript.raw.json`` exclusively.
     11. Optional SRT / VTT.
     12. ``write_manifest`` with run record.
     13. Release lock; clean ``work/`` if ``--no-keep-work`` and success.

    Raises :class:`PipelineError` for user-facing failures; the caller maps
    the error message and exit code into a CLI return code.
    """
    ws = req.workspace
    started_wall = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()

    # Pre-flight: raw.json contract (do this *before* taking the lock so the
    # error message arrives early on a cleanly-resumed run).
    _ensure_raw_json_writable(req)

    acquire_lock(ws)
    forward_stderr = req.json_events

    warnings: list[str] = []
    success = False

    try:
        with EventMirror(ws) as em:
            _emit_event(
                em,
                {
                    "event": "start",
                    "stage": "pipeline",
                    "input": str(req.input_path),
                    "workdir": str(ws.root),
                    "voxkit_version": __version__,
                    "started_at": started_iso,
                },
                forward_to_stderr=forward_stderr,
            )

            # 1. Discovery
            whisper_bin, model_path, vad_model = _discover_binaries(req)
            _emit_event(
                em,
                {
                    "event": "discover",
                    "whisper_cli": str(whisper_bin),
                    "model": str(model_path),
                    "vad_model": str(vad_model) if vad_model else None,
                },
                forward_to_stderr=forward_stderr,
            )
            if req.vad and vad_model is None:
                warnings.append("VAD model not found — proceeding without VAD")

            # 2. Master wav + duration
            master_wav, duration_secs = _prepare_master_wav(
                req, em, forward_stderr=forward_stderr
            )

            # 3. Chunk plan
            # Thresholds 默认走模块常量；env vars (VOXKIT_CHUNK_*) 覆盖供 A/B 诊断。
            threshold_secs, chunk_secs, overlap_secs = chunk_thresholds_from_env()
            plan: ChunkPlan = plan_chunks(
                duration_secs,
                ws.work,
                threshold_secs=threshold_secs,
                chunk_secs=chunk_secs,
                overlap_secs=overlap_secs,
            )
            _emit_event(
                em,
                {
                    "event": "plan",
                    "chunk_count": len(plan.chunks),
                    "thresholds": {
                        "threshold_secs": threshold_secs,
                        "chunk_secs": chunk_secs,
                        "overlap_secs": overlap_secs,
                    },
                    "total_secs": plan.total_secs,
                },
                forward_to_stderr=forward_stderr,
            )

            # 4. Per-chunk transcribe + segment.
            # Load the hallucination blocklist once; the JSON parse + frozenset
            # construction is non-trivial and the blocklist is immutable.
            blocklist = load_blocklist(req.blocklist_path)

            chunk_results: list[ChunkResult] = []
            chunk_stats: list[ChunkStat] = []
            detected_languages: list[str] = []
            inferred_languages: list[str] = []
            total_drops = 0

            for spec in plan.chunks:
                kept, dropped, elapsed, cached, detected_language = _transcribe_chunk(
                    req,
                    spec,
                    whisper_bin=whisper_bin,
                    model_path=model_path,
                    vad_model=vad_model,
                    blocklist=blocklist,
                    em=em,
                    forward_stderr=forward_stderr,
                )
                chunk_language = req.language
                if req.language == "auto":
                    if detected_language is not None:
                        detected_languages.append(detected_language)
                        chunk_language = detected_language
                    else:
                        mode = detect_mode(kept)
                        inferred_language = (
                            "en" if mode == "english_word" else "zh"
                        )
                        inferred_languages.append(inferred_language)
                        chunk_language = inferred_language

                # Segment is chunk-relative (segment.start measured from chunk 0)
                chunk_segments = segment_entries(kept, language=chunk_language)
                chunk_results.append(
                    ChunkResult(
                        chunk_index=spec.index,
                        segments=chunk_segments,
                        chunk_start_secs=spec.start_secs,
                    )
                )
                rtf = (
                    elapsed / spec.duration_secs
                    if spec.duration_secs > 0
                    else 0.0
                )
                chunk_stats.append(
                    ChunkStat(
                        index=spec.index,
                        startSecs=spec.start_secs,
                        durationSecs=spec.duration_secs,
                        elapsedSecs=elapsed,
                        rtf=rtf,
                        cached=cached,
                    )
                )
                total_drops += len(dropped)
                _emit_event(
                    em,
                    {
                        "event": "chunk.done",
                        "chunk": spec.index,
                        "elapsed_secs": elapsed,
                        "cached": cached,
                        "entries_kept": len(kept),
                        "entries_dropped": len(dropped),
                        "segments": len(chunk_segments),
                    },
                    forward_to_stderr=forward_stderr,
                )

            # 5. Merge chunks
            merged_segments, merge_notes = merge_chunks(chunk_results)

            try:
                write_merge_log(chunk_results, merged_segments, None, ws.merge_log)
            except OSError:
                pass  # log write failure is non-fatal

            _emit_event(
                em,
                {
                    "event": "merge.done",
                    "segments": len(merged_segments),
                    "notes": len(merge_notes),
                },
                forward_to_stderr=forward_stderr,
            )

            for note in merge_notes:
                warnings.append(
                    f"merge note: {note.kind} at {note.seg_id} ({note.detail})"
                )

            # 6. Compose TranscriptionOutput
            elapsed_secs = time.monotonic() - started_wall
            rtf_total = (
                elapsed_secs / duration_secs if duration_secs > 0 else 0.0
            )
            language_for_output = _resolve_auto_language(
                req.language,
                detected_languages,
                inferred_languages,
                warnings,
            )

            # Compute these once and reuse across the on-disk dict, the Remixr
            # metadata, and the manifest so the three artifacts stay in sync.
            asr_model_name = model_path.name if model_path else req.model
            word_ts_effective = (
                req.word_timestamps and language_for_output not in CJK_LANGUAGES
            )
            chunk_stats_dump = [c.model_dump(by_alias=True) for c in chunk_stats]

            audio_info = _build_audio_info(req, duration_secs)
            voxkit_out = TranscriptionOutput(
                schemaVersion="1",
                audio=audio_info,
                asrBackend="whisper-cpp",
                asrModel=asr_model_name,
                language=language_for_output,
                wordTimestamps=word_ts_effective,
                rtf=rtf_total,
                elapsedSecs=elapsed_secs,
                perChunk=chunk_stats,
                hallucinationDrops=total_drops,
                segments=merged_segments,
                warnings=warnings,
            )

            # 7. Map → Remixr (in-memory; speaker labels potentially injected
            #    by the diarization pass below before we serialise both the
            #    voxkit-native and Remixr transcripts).
            #
            #    Note: transcript.voxkit.json is written AFTER the diarization
            #    pass so any unmatched-segment warning is captured. The Remixr
            #    raw.json is also written after for the same reason.
            remixr_t = to_remixr_transcript(voxkit_out, source_id=req.source_id)

            # 8b. Phase 2 — optional diarization integration.
            #     Runs AFTER ASR merging so we have the absolute timeline that
            #     the speaker assignment needs.
            diarization_output: DiarizationOutput | None = None
            diarize_elapsed: float | None = None
            unmatched_count = 0
            num_speakers: int | None = None
            if req.with_diarization:
                diarization_output, diarize_elapsed = _run_diarization_pass(
                    req,
                    master_wav=master_wav,
                    duration_secs=duration_secs,
                    em=em,
                    forward_stderr=forward_stderr,
                )
                num_speakers = diarization_output.num_speakers
                _emit_event(
                    em,
                    {
                        "event": "diarize.done",
                        "speakers": num_speakers,
                        "elapsed_secs": diarize_elapsed,
                    },
                    forward_to_stderr=forward_stderr,
                )

                # Audit artefact: the chunk-of-truth diarization JSON lives
                # under work/ alongside whisper checkpoints.
                diarization_audit_path = ws.work / "diarization.json"
                diarization_audit_path.write_text(
                    diarization_output.model_dump_json(by_alias=True, indent=2)
                    + "\n",
                    encoding="utf-8",
                )

                # Speaker assignment via the pure helper.
                from voxkit.core.align_speakers import (
                    SpeakerLabelMode,
                    assign_speakers,
                )
                speaker_labels_mode: SpeakerLabelMode = (
                    "raw" if req.speaker_labels == "raw" else "ranked"
                )
                speaker_by_id, unmatched_ids = assign_speakers(
                    voxkit_out.segments,
                    diarization_output,
                    speaker_labels=speaker_labels_mode,
                    min_dia_duration_s=DIA_PHANTOM_FILTER_S,
                    fallback_to_nearest=True,
                )
                unmatched_count = len(unmatched_ids)
                matched_count = len(voxkit_out.segments) - unmatched_count
                _emit_event(
                    em,
                    {
                        "event": "align.done",
                        "matched": matched_count,
                        "unmatched": unmatched_count,
                    },
                    forward_to_stderr=forward_stderr,
                )

                # Inject labels into the in-memory RemixrTranscript. The
                # voxkit-native schema's ``TranscriptSegment`` deliberately has
                # no ``speaker`` field; the Remixr-shaped output owns speaker
                # identity. The id mapping uses the original
                # ``voxkit_out.segments[i].id`` because
                # ``to_remixr_transcript`` re-numbers on output and we built
                # ``speaker_by_id`` against the voxkit-native ids — so we walk
                # them in lockstep.
                for vox_seg, remixr_seg in zip(
                    voxkit_out.segments, remixr_t.segments
                ):
                    label = speaker_by_id.get(vox_seg.id)
                    if label is not None:
                        remixr_seg.speaker = label
                    else:
                        remixr_seg.speaker = "Speaker ?"

                if unmatched_count > 0:
                    note = (
                        f"alignment: {unmatched_count} segments had no "
                        f"diarization overlap"
                    )
                    warnings.append(note)
                    # Pydantic v2 makes a copy of list inputs at validate-time,
                    # so we need to mutate the model's own list to keep the
                    # transcript.voxkit.json warnings field in sync with the
                    # outer ``warnings`` list used by manifest + raw.json.
                    voxkit_out.warnings.append(note)

            # 8c. Write transcript.voxkit.json (camelCase via aliases).
            #
            # We attach ``sourceId`` to the on-disk dict (not the Pydantic
            # model) because the voxkit-native schema deliberately keeps
            # source identity separate from the rich transcript payload —
            # but downstream consumers + integration tests want to round-trip
            # it without re-reading the manifest. ``model_config`` allows
            # extra keys; this is forward-safe.
            voxkit_payload = voxkit_out.model_dump(by_alias=True, exclude_none=False)
            voxkit_payload["sourceId"] = req.source_id
            ws.voxkit_json_path.write_text(
                json.dumps(voxkit_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _emit_event(
                em,
                {"event": "write.transcript_voxkit", "path": str(ws.voxkit_json_path)},
                forward_to_stderr=forward_stderr,
            )

            # 8d. Now write transcript.raw.json with whatever speaker labels
            #     ended up on it (diarization-injected or "Speaker A" default).
            metadata = {
                "voxkitVersion": __version__,
                "asrBackend": "whisper-cpp",
                "asrModel": asr_model_name,
                "rtf": rtf_total,
                "elapsedSecs": elapsed_secs,
                "language": language_for_output,
                "hallucinationDrops": total_drops,
                "perChunk": chunk_stats_dump,
                "warnings": warnings,
            }
            if diarization_output is not None:
                metadata["withDiarization"] = True
                metadata["speakerLabels"] = req.speaker_labels
                metadata["diarizationModel"] = diarization_output.model
                metadata["diarizationDevice"] = diarization_output.device
                metadata["diarizationElapsedSecs"] = diarize_elapsed
                metadata["numSpeakers"] = num_speakers
            try:
                write_remixr_json(remixr_t, ws.raw_json_path, metadata=metadata)
            except FileExistsError as exc:
                # Should be impossible thanks to _ensure_raw_json_writable, but
                # keep a defensive arm for races on the same workdir.
                raise PipelineError(
                    f"{ws.raw_json_path} already exists (race); pick a fresh --workdir",
                    exit_code=int(ExitCode.GENERIC_FAIL),
                ) from exc
            _emit_event(
                em,
                {"event": "write.transcript_raw", "path": str(ws.raw_json_path)},
                forward_to_stderr=forward_stderr,
            )

            # 9. Subtitles. resegment only affects SRT/VTT — transcript.raw.json
            #    above is ASR ground truth, byte-identical regardless of this flag.
            #    The semantic-resegmented cues additionally land in
            #    subtitles.cues.json (machine-readable mirror of the cue stream)
            #    so downstream consumers do not need to reverse-parse SRT text.
            artifacts: dict[str, Path] = {
                "raw_json": ws.raw_json_path,
                "voxkit_json": ws.voxkit_json_path,
                "manifest": ws.manifest_path,
                "events": ws.events_path,
            }
            if diarization_output is not None:
                artifacts["diarization_json"] = ws.work / "diarization.json"

            cues: "list[SubtitleCue] | None" = None
            cues_from_resegment = False
            resegment_params_snapshot: "dict | None" = None
            if req.resegment == "semantic" and (req.emit_srt or req.emit_vtt):
                try:
                    from dataclasses import asdict
                    from voxkit.core.semantic_resegment import (
                        ResegmentParams,
                        resegment_for_subtitles,
                    )
                    rp = ResegmentParams()
                    cues = resegment_for_subtitles(
                        remixr_t.segments,
                        language=language_for_output,
                        params=rp,
                    )
                    cues_from_resegment = True
                    resegment_params_snapshot = asdict(rp)
                    _emit_event(
                        em,
                        {
                            "event": "resegment.done",
                            "input_segments": len(remixr_t.segments),
                            "output_cues": len(cues),
                        },
                        forward_to_stderr=forward_stderr,
                    )
                except ImportError as exc:
                    msg = (
                        f"resegment=semantic requested but pysbd not available "
                        f"({exc}); falling back to legacy renderer"
                    )
                    warnings.append(msg)
                    voxkit_out.warnings.append(msg)
                    cues = None
                    cues_from_resegment = False

            # Render-layer machine-readable mirror: only when cues came from the
            # semantic resegmenter. The diarized 1-cue-per-segment fallback is a
            # typed bridge — emitting it here would mislabel ASR segments as
            # "semantic" cues.
            if cues_from_resegment and cues is not None:
                from voxkit.io.cues_json import write_cues_json
                try:
                    write_cues_json(
                        cues,
                        ws.cues_json_path,
                        source_id=req.source_id,
                        resegment="semantic",
                        params=resegment_params_snapshot,
                    )
                except FileExistsError as exc:
                    raise PipelineError(
                        f"{ws.cues_json_path} already exists (race); "
                        f"pick a fresh --workdir",
                        exit_code=int(ExitCode.GENERIC_FAIL),
                    ) from exc
                artifacts["subtitle_cues_json"] = ws.cues_json_path
                _emit_event(
                    em,
                    {
                        "event": "write.subtitle_cues",
                        "path": str(ws.cues_json_path),
                        "cue_count": len(cues),
                    },
                    forward_to_stderr=forward_stderr,
                )

            # Diarized path collapses to the cue renderer too: adapt
            # RemixrSegment → SubtitleCue (1:1) so SRT/VTT have a single
            # implementation regardless of resegment / diarization combination.
            if cues is None and diarization_output is not None:
                cues = _remixr_to_cues(remixr_t)

            if req.emit_srt:
                if cues is not None:
                    from voxkit.io.srt import to_subtitles_srt_from_cues
                    srt_text = to_subtitles_srt_from_cues(cues)
                else:
                    srt_text = to_subtitles_srt(voxkit_out)
                ws.srt_path.write_text(srt_text, encoding="utf-8")
                artifacts["srt"] = ws.srt_path
                _emit_event(
                    em,
                    {"event": "write.srt", "path": str(ws.srt_path)},
                    forward_to_stderr=forward_stderr,
                )
            if req.emit_vtt:
                if cues is not None:
                    from voxkit.io.srt import to_subtitles_vtt_from_cues
                    vtt_text = to_subtitles_vtt_from_cues(cues)
                else:
                    vtt_text = to_subtitles_vtt(voxkit_out)
                ws.vtt_path.write_text(vtt_text, encoding="utf-8")
                artifacts["vtt"] = ws.vtt_path
                _emit_event(
                    em,
                    {"event": "write.vtt", "path": str(ws.vtt_path)},
                    forward_to_stderr=forward_stderr,
                )

            # 10. Manifest (single source of truth for run audit)
            manifest = {
                "voxkitVersion": __version__,
                "schemaVersion": "1",
                "startedAt": started_iso,
                "finishedAt": datetime.now(timezone.utc).isoformat(),
                "input": str(req.input_path),
                "sourceId": req.source_id,
                "workdir": str(ws.root),
                "asrBackend": "whisper-cpp",
                "asrModel": asr_model_name,
                "language": language_for_output,
                "wordTimestamps": word_ts_effective,
                "vad": bool(req.vad and vad_model is not None),
                "vadModel": str(vad_model) if vad_model else None,
                "whisperBin": str(whisper_bin),
                "logprobThold": req.logprob_thold,
                "resume": req.resume,
                "elapsedSecs": elapsed_secs,
                "rtf": rtf_total,
                "durationSecs": duration_secs,
                "chunkCount": len(plan.chunks),
                "chunkThresholds": {
                    "thresholdSecs": threshold_secs,
                    "chunkSecs": chunk_secs,
                    "overlapSecs": overlap_secs,
                },
                "perChunk": chunk_stats_dump,
                "hallucinationDrops": total_drops,
                "mergeNotes": [
                    {"kind": n.kind, "segId": n.seg_id, "detail": n.detail}
                    for n in merge_notes
                ],
                "warnings": warnings,
                "artifacts": {k: str(v) for k, v in artifacts.items()},
                # ── Phase 2: diarization metadata ────────────────────
                "withDiarization": req.with_diarization,
                "speakerLabels": req.speaker_labels,
                "diarizationModel": (
                    diarization_output.model if diarization_output else None
                ),
                "diarizationDevice": (
                    diarization_output.device if diarization_output else None
                ),
                "diarizationElapsedSecs": diarize_elapsed,
                "numSpeakers": num_speakers,
                # ── Phase 3: subtitle resegment metadata ─────────────
                "subtitle": {
                    "resegment": req.resegment,
                    "cueCount": len(cues) if cues is not None else None,
                },
            }
            write_manifest(ws, manifest)
            _emit_event(
                em,
                {
                    "event": "done",
                    "elapsed_secs": elapsed_secs,
                    "rtf": rtf_total,
                    "segments": len(merged_segments),
                },
                forward_to_stderr=forward_stderr,
            )

            success = True
            return TranscribeResult(
                voxkit_output=voxkit_out,
                artifacts=artifacts,
                warnings=warnings,
                elapsed_secs=elapsed_secs,
                rtf=rtf_total,
            )
    finally:
        try:
            release_lock(ws)
        except Exception:  # noqa: BLE001 — best-effort
            pass
        # Cleanup work/ if user asked + run actually succeeded.
        if success and not req.keep_work:
            try:
                shutil.rmtree(ws.work)
            except OSError:
                pass

# Changelog

All notable changes to voxkit. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project does NOT follow strict semver until 1.0.0; minor versions may include breaking
changes (with migration notes).

## [Unreleased]

### Added

- **`subtitles.cues.json`** — render-layer machine-readable mirror of the
  semantic resegmenter output. Written only when `--resegment=semantic` and
  the resegment path actually produced cues (the diarized 1-cue-per-segment
  fallback is excluded). Schema:
  `{schemaVersion, sourceId, resegment, params, cues[{start,end,speaker,text}]}`.
  Lets downstream consumers (e.g. Remixr) ingest semantic cues directly
  instead of reverse-parsing SRT text. `transcript.raw.json` stays untouched
  — it is ASR ground truth; cues are render-layer derivatives, see
  `docs/transcribe.md` for the rationale.
- New events `write.subtitle_cues` (path + cue_count) and the existing
  `resegment.done` now bracket cues.json emission.
- `Workspace.cues_json_path`; `manifest.artifacts.subtitle_cues_json`
  populated when written.
- `voxkit.io.cues_json` module + `SubtitleCueOut` / `SubtitleCuesOutput`
  Pydantic models in `io/schema.py`.

### Changed

- `_ensure_raw_json_writable` also unlinks `subtitles.cues.json` on `--force`
  so the exclusive-create write does not collide on rerun.

## [0.3.0] — 2026-04-28

This release renames the project from `voxsplit` to `voxkit` and adds a `transcribe` subcommand
backed by whisper.cpp, repositioning voxkit as a "speech-to-structured-data toolkit" rather than
a single-purpose diarization CLI. The whisper.cpp invocation stack (chunking, anti-hallucination,
dual-mode segmentation, overlap dedup, checkpoint resume) is ported from Remixr's TypeScript
implementation, replacing ~1400 lines of `services/whisper.ts` with a Python equivalent that any
caller can hit via the CLI.

### Added

- **`voxkit transcribe`** subcommand — whisper.cpp ASR with the full anti-hallucination stack:
  - 3-layer defense: VAD (silero) + `--max-context 0` + `--logprob-thold -0.8`.
  - Long-audio chunking: 15-min threshold, 10-min chunks, 5s overlap, 0.5s overlap-dedup
    tolerance.
  - Word-level timestamps in English mode (`--max-len 1 --split-on-word`); CJK languages
    (`zh` / `ja` / `yue` / `ko`) auto-drop those flags and emit phrase-level segments with
    empty `words[]`.
  - Chinese hallucination blocklist (7 watermark prefixes + 19 standalone phrases + ghost CJK
    loop detection: ≥6-char CJK substring repeated ≥2 times).
  - NFC normalization before blocklist matching (whisper.cpp occasionally emits NFD).
  - Checkpoint resume via `work/chunks/chunk_NNN.json` cache; `--force` clears `work/`.
  - Dynamic per-chunk timeout: `max(30 min, duration * 0.3)`.
  - VAD model 3-level fallback: `--vad-model` flag → `WHISPER_VAD_MODEL_PATH` env →
    `/opt/homebrew/share/whisper-cpp/ggml-silero-v5.1.2.bin`; warn-once and disable VAD if
    none found.
  - whisper-cli discovery: `--whisper-bin` → `$WHISPER_BIN` → `which whisper-cli` →
    `/opt/homebrew/bin/whisper-cli`.
- **Workdir-based artifact layout** — data orthogonal, fully auditable, concurrent-safe via
  per-workdir PID lock:
  - `transcript.raw.json` — Remixr Zod-compatible (drop-in for
    `storage/projects/{projectId}/sources/{sourceId}/transcript.raw.json`); written with
    exclusive `wx` mode (re-running the same workdir errors loudly).
  - `transcript.voxkit.json` — rich native format with RTF, elapsed, perChunk stats,
    hallucinationDrops, warnings.
  - `subtitles.srt` + `subtitles.vtt` — segment-level cues (`--emit-srt` / `--emit-vtt`,
    both on by default).
  - `manifest.json` — input, args, voxkit version, whisper-cli version, start/end times,
    PID lock.
  - `events.ndjson` — mirror of stderr NDJSON event stream (always written).
  - `work/input.16khz.mono.wav` — ffmpeg-normalized master (`-ar 16000 -ac 1`).
  - `work/chunks/chunk_NNN.{wav,json,entries.json}` — per-chunk audio + raw whisper output +
    post-blocklist filtered entries.
  - `work/hallucinations.log` — NDJSON record of every dropped entry + reason code.
  - `work/merge.json` — per-chunk segment-id keep/drop decisions.
  - `work/timeline_validation.log` — warn-only timeline-continuity check output.
- **Remixr adapter** (`io/remixr_adapter.py`) — single point of truth for `transcript.raw.json`
  schema mapping. Embeds optional `_metadata` (voxkit version, asrBackend, asrModel, language,
  sourceDurationSecs, processedAt, whisperBin, vadModel, RTF, perChunk, warnings); Remixr
  ignores unknown fields, safe for audit.
- **SRT/VTT generators** (`io/srt.py`) — segment-level cues, `"Speaker A: "` placeholder until
  diarization is chained (Phase 2).
- **`Workspace`** (`core/workspace.py`) — frozen dataclass exposing every workdir path +
  `EventMirror` context manager (tees NDJSON to `events.ndjson`) + PID lock via independent
  `<workdir>/.lock` file (`O_CREAT | O_EXCL` atomic create); stale-PID detection downgrades
  to warning.
- **Doctor checks** (3 new) — all WARN-only so diarize-only users don't regress:
  - `check_whisper_cli()` — verifies `whisper-cli` on PATH and greps required flags
    (`--output-json-full`, `--max-context`, `--vad`, `--split-on-word`, `--logprob-thold`).
  - `check_whisper_model()` — discovers the configured ggml model.
  - `check_vad_model()` — discovers the silero VAD bin.
- **Bundle aux files** — `BUNDLE_AUX_FILES` extends bundle to include the silero VAD bin
  (~885 KB). The whisper.cpp ggml model itself stays OUT (license boundary + 1.5 GB size);
  `voxkit doctor` directs users to `brew install whisper-cpp` and
  `huggingface-cli download ggerganov/whisper.cpp ggml-large-v3-turbo.bin`.
- **Pydantic models** (`io/schema.py` extended): `TranscriptionOutput`, `TranscriptSegment`,
  `Word`, `ChunkStat`, `RemixrTranscript`, `RemixrSegment`, `RemixrWord`. camelCase aliases
  via `populate_by_name=True`. New `TranscriptionOutput.schemaVersion = "1"` (independent
  counter from `DiarizationOutput.schemaVersion`).
- **Internal `Entry` type** (`core/types.py`) — frozen dataclass for whisper.cpp transcription
  rows; bridge between `whisper_exec` / `segmenter` / `hallucination_filter`.
- **Pipeline orchestration** (`core/transcribe_pipeline.py`) — `run_pipeline(req, progress)`
  drives audio prep → chunk plan → per-chunk whisper (resume-aware) → blocklist filter →
  segmenter → ASR merge → write voxkit / raw / SRT / VTT / manifest.
- **Audio extensions** (`core/audio.py`):
  - `plan_chunks(duration, work_dir, *, threshold=900, chunk=600, overlap=5)` — chunk plan
    builder.
  - `normalize_to_wav_16k_mono(input, out_wav)` — ffmpeg normalization.
  - `extract_chunk(master_wav, spec)` — `-ss` before `-i` for accurate input-seek.
- **Dual-mode segmenter** (`core/segmenter.py`) — `detect_mode()` uses leading-space ratio
  ≥0.5 to pick English word mode vs Chinese phrase mode (`--language` overrides). 4-priority
  segment boundary: punctuation-end > 500 ms gap > 5 s duration > 100 chars.
- **ASR merge** (`core/asr_merge.py`) — overlap-dedup + offset.
  `offset_segment(seg, delta)` shifts BOTH `segment.{start,end}` AND `words[].{start,end}` —
  this synchronization is the regression fix for a 6-month bug in Remixr's TS implementation,
  enforced by a hard regression test in `tests/test_asr_merge.py`.
- **Hallucination filter** (`core/hallucination_filter.py` + `data/hallucination_blocklist.yaml`)
  — three rules in order: watermark prefix (NFC-normalized startswith), standalone exact match,
  ghost CJK loop. Drops are logged as NDJSON to `hallucinations.log`.
- **Whisper exec** (`core/whisper_exec.py`) — pure-function `build_argv()` (snapshot-testable),
  `Popen`-based `run_whisper()` that streams stderr and parses `progress\s*=\s*(\d+)%` to emit
  `{event: "progress", stage: "whisper.chunk", chunk, percent}`.
- 200+ new unit + integration tests (`test_transcribe_e2e` gated by `requires_whisper`
  marker for environments without whisper.cpp installed).

### Changed

- **BREAKING — package rename**: `voxsplit` → `voxkit`. All Python imports change.
- **BREAKING — CLI rename**: `voxsplit` command → `voxkit`. The old name is NOT aliased;
  user must update scripts.
- **BREAKING — user data paths**:
  - `~/.local/share/voxsplit/venv` → `~/.local/share/voxkit/venv`
  - `~/.cache/voxsplit/.installed` → `~/.cache/voxkit/.installed`
- **BREAKING — bundle filename**: `voxsplit-models.tar.gz` → `voxkit-models.tar.gz`,
  `voxsplit-models.manifest.json` → `voxkit-models.manifest.json`.
- **BREAKING — bundle GitHub repo**: `3Craft/voxsplit` → `3Craft/voxkit`. The old repo
  retains v0.2.x bundles for archival; v0.3.0+ bundles publish to the new repo only.
- **Worker subprocess module path**: `python -m voxsplit.core.pipeline` →
  `python -m voxkit.core.pipeline`.
- **Worker stdout sentinel**: `__VOXSPLIT_JSON__` → `__VOXKIT_JSON__`.
- **`pyproject [project.scripts]` entry**: `voxsplit = "voxsplit.cli:main"` →
  `voxkit = "voxkit.cli:main"`.
- **`cli.py` `prog=`**: `prog="voxsplit"` → `prog="voxkit"`.

### Stable / non-breaking

- **`DiarizationOutput.schemaVersion` stays `"1"`** — existing JSON consumers do not need
  updates.
- All v0.2.x subcommands preserved with identical CLI contracts (post-rename of the prog name):
  `diarize`, `align`, `doctor`, `setup`, `build-bundle`, `fetch-bundle`.
- Worker venv lazy-install mechanism (`uv venv` + `pyannote.audio>=4.0.4,<5`) unchanged;
  whisper.cpp + ffmpeg + silero VAD model stay OUT of venv (native binaries discovered via
  PATH).
- `commands/align.py` already reads voxkit-style transcript JSON (`segments[].start/end/text`);
  no logic change needed.

### Compatibility & dependencies

- whisper.cpp 1.7+ recommended (verified against 1.8.4); requires flags
  `--output-json-full`, `--max-context`, `--vad`, `--split-on-word`, `--logprob-thold`.
  `voxkit doctor` warn-fails (non-fatal) if any are missing.
- Default whisper model: `large-v3-turbo` (FP16, 1.5 GB). Alternative:
  `large-v3-turbo-q5_0` (547 MB, slightly lower quality). User installs via `brew` or
  `huggingface-cli`.
- pyannote / torch remain in the lazy-install venv (only used by `diarize`).

### Migration notes

For users upgrading from voxsplit 0.2.x:

```bash
# 1. Rename user data (or just re-run `voxkit setup` to recreate fresh).
mv ~/.local/share/voxsplit ~/.local/share/voxkit
mv ~/.cache/voxsplit ~/.cache/voxkit

# 2. Reinstall the new entrypoint (uv tool / pipx / pip — pick one).
uv tool install --force voxkit
# or: pipx install --force voxkit
# or: pip install --upgrade voxkit

# 3. Fetch the v0.3.0+ bundle from the new repo.
voxkit fetch-bundle              # pulls from 3Craft/voxkit, not 3Craft/voxsplit

# 4. (For transcribe) Install whisper.cpp + the ggml model.
brew install whisper-cpp ffmpeg-full
huggingface-cli download ggerganov/whisper.cpp ggml-large-v3-turbo.bin \
  --local-dir ~/.cache/voxkit/models

# 5. Verify.
voxkit doctor                    # 10 checks (7 inherited + 3 new whisper-related)
```

Existing `DiarizationOutput` JSON consumers do not need code changes — `schemaVersion` is
still `"1"`. Scripts referencing the `voxsplit` command must be updated to `voxkit`.

### Known issues

None known at release.

### Verification

- 263+ tests passing (`pytest tests/ -q`).
- End-to-end real-world test: 64-min English podcast → 7 chunks, RTF 0.0476, 909 segments,
  0 hallucinations, full Remixr Zod conformance.
- Concurrency: two `voxkit transcribe` runs against different workdirs do not interfere
  (data orthogonal); same workdir is rejected by `wx` exclusive write on
  `transcript.raw.json`.
- Resume: re-running `voxkit transcribe` on an existing workdir keeps `chunk_NNN.json`
  mtimes unchanged; `--force` updates them all.

## [0.2.x] — voxsplit (archived)

Released as the `voxsplit` package. See the `3Craft/voxsplit` GitHub repo for v0.2.x release
history. voxkit 0.3.0 is a renamed continuation; the functional content of voxsplit 0.2.x
(`diarize`, `align`, `doctor`, `setup`, `build-bundle`, `fetch-bundle` commands) is preserved
verbatim.

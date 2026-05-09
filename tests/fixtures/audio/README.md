# Audio Fixtures

Small (~5s, 16kHz, mono, PCM 16-bit) audio files for end-to-end tests.

| File | Generation | Content | Used by |
|---|---|---|---|
| `short_en.wav` | ffmpeg sine 440Hz 5s | tone (placeholder) | tests/test_transcribe_e2e.py::test_e2e_english_pipeline |
| `short_zh.wav` | ffmpeg sine 660Hz 5s | tone (placeholder) | tests/test_transcribe_e2e.py::test_e2e_chinese_pipeline |

## Generation commands

```bash
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 -ac 1 -c:a pcm_s16le short_en.wav
ffmpeg -y -f lavfi -i "sine=frequency=660:duration=5" -ar 16000 -ac 1 -c:a pcm_s16le short_zh.wav
```

Each file is ~156KB, well under the 200KB target.

## What we are (and are not) testing

These fixtures test PIPELINE INTEGRATION, not ASR quality. Whisper will produce
empty or noise transcripts on pure sine waves; the test asserts that the
end-to-end pipeline COMPLETES (writes all artifacts, exits 0) — not that the
transcript is correct.

The Chinese fixture (`short_zh.wav`) carries a `_zh` label only — its content
is identical-format placeholder audio. The label drives the `language="zh"`
code path through the segmenter (chinese_phrase mode) and the CJK branch in
`whisper_exec` (no `--max-len 1 --split-on-word`). It exercises format
compatibility for CJK pipelines, not transcription accuracy.

For ASR quality verification, use the dry-run scripts in `tmp/` (see
`tmp/dryrun/README.md`).

## Why not reuse `tests/fixtures/short.wav`?

The pre-existing `tests/fixtures/short.wav` is a 60s recording — too long for
fast e2e tests where we run whisper-cli end-to-end in CI/local matrix. We
purposely keep these new fixtures at 5s so the gated `requires_whisper` tests
stay snappy when whisper-cli + a model are installed.

## Synthetic boundary fixtures

`build_synthetic_audio.py` generates deterministic WAV files for segmentation
experiments. These files are not committed by default because scaled variants
can get large.

```bash
python tests/fixtures/audio/build_synthetic_audio.py
```

Default output goes to `tests/fixtures/audio/generated/` and uses a short
12s chunk target:

| Scenario | Purpose |
|---|---|
| `boundary_silence_near_target.wav` | silence candidates around the target boundary |
| `boundary_no_silence.wav` | fallback when the search window contains no silence |
| `boundary_multi_candidate.wav` | scoring tradeoff between distance and silence duration |
| `cjk_phrase_cadence.wav` | cadence smoke fixture for CJK subtitle resegmentation experiments |

Each WAV has a sidecar JSON manifest with suggested test parameters:

- `suggested_chunk_threshold_secs`
- `suggested_chunk_secs`
- `suggested_chunk_overlap_secs`
- `target_boundary_secs`
- segment labels and exact tone/silence intervals

For manual full-scale tests near the production 600s boundary:

```bash
python tests/fixtures/audio/build_synthetic_audio.py \
  --out-dir /tmp/voxkit-synthetic-audio \
  --scale 50
```

The short default scenarios are the ones intended for CI. Full-scale variants
are useful for listening tests and end-to-end timing experiments.

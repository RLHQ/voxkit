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

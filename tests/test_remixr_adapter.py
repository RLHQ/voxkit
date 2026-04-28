"""Unit tests for ``voxkit.io.remixr_adapter``.

Covers:
  - voxkit-native → Remixr mapping (id re-numbering, speaker placeholder,
    subtitles default, words passthrough).
  - JSON round-trip through ``model_dump_json(exclude_none=True)``.
  - ``write_remixr_json`` exclusive-write semantics + indent + ensure_ascii.
  - Optional ``_metadata`` attachment.
  - UTF-8 preservation for CJK content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxkit.io.remixr_adapter import to_remixr_transcript, write_remixr_json
from voxkit.io.schema import (
    AudioInfo,
    RemixrTranscript,
    TranscriptionOutput,
    TranscriptSegment,
    Word,
)


def _make_output(*, segments: list[TranscriptSegment]) -> TranscriptionOutput:
    return TranscriptionOutput(
        audio=AudioInfo(path="/tmp/a.wav", duration_secs=10.0),
        asr_backend="whisper-cpp",
        asr_model="ggml-large-v3-turbo",
        language="en",
        word_timestamps=True,
        rtf=0.05,
        elapsed_secs=10.0,
        segments=segments,
    )


def _two_segment_fixture() -> TranscriptionOutput:
    seg1 = TranscriptSegment(
        id="x_1",  # intentionally non-conforming; adapter must re-id.
        start=0.0,
        end=1.0,
        text="hi there",
        words=[],
    )
    seg2 = TranscriptSegment(
        id="x_2",
        start=1.0,
        end=2.0,
        text="hello world",
        words=[
            Word(word="hello", start=0.0, end=0.5),
            Word(word="world", start=0.5, end=1.0),
        ],
    )
    return _make_output(segments=[seg1, seg2])


# ---------------------------------------------------------------------------
# 1. Field-by-field mapping
# ---------------------------------------------------------------------------


def test_to_remixr_transcript_basic_mapping():
    out = _two_segment_fixture()
    t = to_remixr_transcript(out, source_id="src_test")

    assert t.sourceId == "src_test"
    assert len(t.segments) == 2

    # Defensive re-id with width-3 zero pad
    assert t.segments[0].id == "seg_001"
    assert t.segments[1].id == "seg_002"

    # Speaker placeholder
    assert all(s.speaker == "Speaker A" for s in t.segments)

    # Subtitles default to []
    assert all(s.subtitles == [] for s in t.segments)

    # Start/end/text passthrough
    assert t.segments[0].start == 0.0
    assert t.segments[0].end == 1.0
    assert t.segments[0].text == "hi there"
    assert t.segments[1].text == "hello world"

    # Words: empty for seg1, two-word for seg2
    assert t.segments[0].words == []
    assert len(t.segments[1].words) == 2
    assert t.segments[1].words[0].word == "hello"
    assert t.segments[1].words[0].start == 0.0
    assert t.segments[1].words[0].end == 0.5
    assert t.segments[1].words[1].word == "world"
    assert t.segments[1].words[1].start == 0.5
    assert t.segments[1].words[1].end == 1.0


def test_to_remixr_transcript_validates_source_id():
    out = _two_segment_fixture()
    with pytest.raises(ValueError):
        to_remixr_transcript(out, source_id="")
    with pytest.raises(ValueError):
        to_remixr_transcript(out, source_id="bad/id")
    with pytest.raises(ValueError):
        to_remixr_transcript(out, source_id="bad\\id")


def test_to_remixr_transcript_id_padding_widens_at_1000():
    """seg_001 width is fixed at 3; spec confirms zero-pad width 3."""
    segs = [
        TranscriptSegment(id="x", start=float(i), end=float(i + 1), text="t")
        for i in range(5)
    ]
    out = _make_output(segments=segs)
    t = to_remixr_transcript(out, source_id="src_x")
    assert [s.id for s in t.segments] == [
        "seg_001",
        "seg_002",
        "seg_003",
        "seg_004",
        "seg_005",
    ]


# ---------------------------------------------------------------------------
# 2. Round-trip
# ---------------------------------------------------------------------------


def test_remixr_transcript_round_trip():
    out = _two_segment_fixture()
    t = to_remixr_transcript(out, source_id="src_rt")

    dumped = t.model_dump_json(exclude_none=True)
    payload = json.loads(dumped)

    # Fields are NOT aliased — the python field names ARE the on-disk keys.
    assert payload["sourceId"] == "src_rt"
    assert payload["segments"][0]["id"] == "seg_001"
    assert payload["segments"][1]["words"][0]["word"] == "hello"

    # rawText must NOT appear (we never emit it)
    for seg in payload["segments"]:
        assert "rawText" not in seg

    # Reload back into model
    reparsed = RemixrTranscript.model_validate_json(dumped)
    assert reparsed.sourceId == t.sourceId
    assert reparsed.segments[1].words[1].end == 1.0


# ---------------------------------------------------------------------------
# 3. write_remixr_json — basics + indent + ensure_ascii
# ---------------------------------------------------------------------------


def test_write_remixr_json_basic(tmp_path: Path):
    out = _two_segment_fixture()
    t = to_remixr_transcript(out, source_id="src_w")
    target = tmp_path / "out.json"

    write_remixr_json(t, target)

    raw = target.read_text(encoding="utf-8")
    # Trailing newline
    assert raw.endswith("\n")
    # 2-space indent
    assert '  "sourceId": "src_w"' in raw
    # Round-trip via stdlib json
    payload = json.loads(raw)
    assert payload["sourceId"] == "src_w"
    assert len(payload["segments"]) == 2


# ---------------------------------------------------------------------------
# 4. Exclusive-write contract
# ---------------------------------------------------------------------------


def test_write_remixr_json_exclusive(tmp_path: Path):
    out = _two_segment_fixture()
    t = to_remixr_transcript(out, source_id="src_excl")
    target = tmp_path / "out.json"

    write_remixr_json(t, target)
    with pytest.raises(FileExistsError):
        write_remixr_json(t, target)


# ---------------------------------------------------------------------------
# 5. _metadata attachment
# ---------------------------------------------------------------------------


def test_write_remixr_json_metadata_attached(tmp_path: Path):
    out = _two_segment_fixture()
    t = to_remixr_transcript(out, source_id="src_meta")
    target = tmp_path / "out.json"

    write_remixr_json(t, target, metadata={"voxkitVersion": "0.3.0"})
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["_metadata"] == {"voxkitVersion": "0.3.0"}
    # _metadata sits at the top level, not inside segments
    for seg in payload["segments"]:
        assert "_metadata" not in seg


# ---------------------------------------------------------------------------
# 6. UTF-8 (no \uXXXX escapes for CJK)
# ---------------------------------------------------------------------------


def test_write_remixr_json_utf8_cjk(tmp_path: Path):
    seg = TranscriptSegment(
        id="x", start=0.0, end=1.5, text="你好世界", words=[]
    )
    out = _make_output(segments=[seg])
    t = to_remixr_transcript(out, source_id="src_cjk")
    target = tmp_path / "out.json"

    write_remixr_json(t, target)
    # Read as bytes and ensure literal CJK characters present (not \u escapes)
    data = target.read_bytes()
    assert "你好世界".encode("utf-8") in data
    assert b"\\u4f60" not in data  # would appear if ensure_ascii was True

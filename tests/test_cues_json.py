"""Unit tests for ``voxkit.io.cues_json`` — render-layer cues serializer.

Covers:
  - ``to_cues_output`` shape: schemaVersion / sourceId / resegment / params / cues
  - ``write_cues_json`` produces parseable UTF-8 JSON with by_alias keys
  - exclusive-create contract: re-writing the same path raises FileExistsError
  - ``params=None`` and ``speaker=None`` are excluded by ``exclude_none``
  - empty cues list serializes cleanly (no ``cues: null``)
  - ensure_ascii=False keeps non-ASCII (CJK) readable in storage
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from voxkit.io.cues_json import to_cues_output, write_cues_json


# Local stand-in for ``voxkit.core.semantic_resegment.SubtitleCue`` so this
# unit test stays decoupled from pysbd import surface.
@dataclass(frozen=True)
class _Cue:
    start: float
    end: float
    speaker: str | None
    text: str


# ---------------------------------------------------------------------------
# to_cues_output
# ---------------------------------------------------------------------------


def test_to_cues_output_basic_shape():
    cues = [
        _Cue(0.10, 5.48, "Speaker A", "Since last year"),
        _Cue(5.50, 9.12, "Speaker B", "Yeah, exactly."),
    ]
    out = to_cues_output(
        cues,
        source_id="abc123",
        resegment="semantic",
        params={"max_dur_s": 7.0},
    )
    assert out.schema_version == "1"
    assert out.source_id == "abc123"
    assert out.resegment == "semantic"
    assert out.params == {"max_dur_s": 7.0}
    assert len(out.cues) == 2
    assert out.cues[0].speaker == "Speaker A"
    assert out.cues[1].text == "Yeah, exactly."


def test_to_cues_output_speaker_none_preserved_at_model_level():
    out = to_cues_output(
        [_Cue(0.0, 1.0, None, "no diarization")],
        source_id="x",
        resegment="semantic",
    )
    assert out.cues[0].speaker is None


# ---------------------------------------------------------------------------
# write_cues_json
# ---------------------------------------------------------------------------


def _roundtrip(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_cues_json_basic(tmp_path: Path):
    cues = [_Cue(0.10, 5.48, "Speaker A", "Hello world")]
    p = tmp_path / "subtitles.cues.json"
    write_cues_json(
        cues,
        p,
        source_id="src1",
        resegment="semantic",
        params={"max_chars": 84},
    )
    data = _roundtrip(p)
    assert data["schemaVersion"] == "1"
    assert data["sourceId"] == "src1"
    assert data["resegment"] == "semantic"
    assert data["params"] == {"max_chars": 84}
    assert data["cues"] == [
        {"start": 0.10, "end": 5.48, "speaker": "Speaker A", "text": "Hello world"}
    ]


def test_write_cues_json_exclusive_create(tmp_path: Path):
    p = tmp_path / "subtitles.cues.json"
    write_cues_json([], p, source_id="s", resegment="semantic")
    with pytest.raises(FileExistsError):
        write_cues_json([], p, source_id="s", resegment="semantic")


def test_write_cues_json_excludes_none(tmp_path: Path):
    """``params=None`` and ``speaker=None`` must not appear as ``null`` keys."""
    cues = [_Cue(0.0, 1.0, None, "no speaker")]
    p = tmp_path / "subtitles.cues.json"
    write_cues_json(cues, p, source_id="s", resegment="semantic", params=None)
    data = _roundtrip(p)
    assert "params" not in data
    assert "speaker" not in data["cues"][0]


def test_write_cues_json_empty_cues(tmp_path: Path):
    p = tmp_path / "subtitles.cues.json"
    write_cues_json([], p, source_id="s", resegment="semantic")
    data = _roundtrip(p)
    assert data["cues"] == []


def test_write_cues_json_unicode_preserved(tmp_path: Path):
    """ensure_ascii=False — CJK should round-trip as readable characters,
    not ``\\uXXXX`` escapes."""
    cues = [_Cue(0.0, 1.0, "讲者 A", "你好，世界")]
    p = tmp_path / "subtitles.cues.json"
    write_cues_json(cues, p, source_id="s", resegment="semantic")
    raw = p.read_text(encoding="utf-8")
    assert "你好，世界" in raw
    assert "讲者 A" in raw


def test_write_cues_json_trailing_newline(tmp_path: Path):
    p = tmp_path / "subtitles.cues.json"
    write_cues_json([], p, source_id="s", resegment="semantic")
    assert p.read_text(encoding="utf-8").endswith("\n")

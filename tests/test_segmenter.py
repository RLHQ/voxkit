"""Tests for :mod:`voxkit.core.segmenter`.

Covers:
  - mode detection (language hint + ratio fallback + edge cases)
  - English word-mode aggregation (4 boundary conditions + EOF)
  - Chinese phrase-mode 1:1 mapping
  - regression sanity vs the prototype dry-run (~900 segs for 11270 entries)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxkit.core.segmenter import (
    CJK_LANGUAGES,
    GAP_MS,
    MAX_CHARS,
    MAX_DUR_MS,
    PUNCT_END_RE,
    detect_mode,
    segment_entries,
)
from voxkit.core.types import Entry

FIXTURES = Path(__file__).parent / "fixtures" / "whisper"
ENGLISH_FIXTURE = FIXTURES / "english_short.json"
CHINESE_FIXTURE = FIXTURES / "chinese_short.json"
DRYRUN_RAW = Path("/Users/xsharp/Workspace/3Craft/voxsplit/tmp/dryrun/whisper-raw.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_entries(fixture_path: Path) -> list[Entry]:
    """Tiny inline parser: whisper.cpp JSON → list[Entry].

    Filters empty / special-token entries (``[_BEG_]`` etc.) the same way
    Agent X's parse_whisper_json will. We inline this to avoid a circular
    import dependency on whisper_exec while it's being written in parallel.
    """
    raw = json.loads(fixture_path.read_text())
    out: list[Entry] = []
    for r in raw["transcription"]:
        text = r["text"]
        stripped = text.strip()
        if not stripped:
            continue
        if stripped.startswith("[_") and stripped.endswith("]"):
            continue
        out.append(
            Entry(
                text=text,
                t_from_ms=int(r["offsets"]["from"]),
                t_to_ms=int(r["offsets"]["to"]),
            )
        )
    return out


def _mk(text: str, t_from: int, t_to: int) -> Entry:
    """Shorthand to build Entry for ad-hoc test cases."""
    return Entry(text=text, t_from_ms=t_from, t_to_ms=t_to)


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


class TestDetectMode:
    def test_empty_entries_default_english(self) -> None:
        assert detect_mode([], None) == "english_word"

    def test_leading_space_entries_english(self) -> None:
        entries = [_mk(" since", 0, 200), _mk(" last", 200, 400)]
        assert detect_mode(entries, None) == "english_word"

    def test_no_leading_space_chinese(self) -> None:
        entries = [_mk("你好", 0, 1000), _mk("世界", 1000, 2000)]
        assert detect_mode(entries, None) == "chinese_phrase"

    def test_ratio_at_threshold_inclusive_english(self) -> None:
        # Exactly 50% leading-space → english (≥ 0.5 inclusive, matching
        # Remixr TS `>= 0.5`).
        entries = [
            _mk(" hi", 0, 100),
            _mk(" yo", 100, 200),
            _mk(" eh", 200, 300),
            _mk(" no", 300, 400),
            _mk("中", 400, 500),
            _mk("文", 500, 600),
            _mk("混", 600, 700),
            _mk("合", 700, 800),
        ]
        assert detect_mode(entries, None) == "english_word"

    def test_language_hint_zh_overrides_english_looking(self) -> None:
        # All entries look English (leading space) but hint says Chinese.
        entries = [_mk(" foo", 0, 100), _mk(" bar", 100, 200)]
        assert detect_mode(entries, "zh") == "chinese_phrase"

    def test_language_hint_en_overrides_chinese_looking(self) -> None:
        entries = [_mk("你好", 0, 1000), _mk("世界", 1000, 2000)]
        assert detect_mode(entries, "en") == "english_word"

    def test_all_cjk_hints_route_to_phrase(self) -> None:
        entries = [_mk(" anything", 0, 100)]  # English-looking
        for hint in CJK_LANGUAGES:
            assert detect_mode(entries, hint) == "chinese_phrase", hint

    def test_unknown_hint_falls_back_to_ratio(self) -> None:
        # Unknown hint "fr" → fall to ratio. Leading-space → english.
        entries = [_mk(" salut", 0, 100), _mk(" tout", 100, 200)]
        assert detect_mode(entries, "fr") == "english_word"

    def test_hint_case_insensitive(self) -> None:
        entries = [_mk(" foo", 0, 100)]
        assert detect_mode(entries, "ZH") == "chinese_phrase"
        assert detect_mode(entries, "EN") == "english_word"


# ---------------------------------------------------------------------------
# English word mode — fixture-based
# ---------------------------------------------------------------------------


class TestEnglishMode:
    def test_fixture_segments_nontrivial(self) -> None:
        entries = _load_entries(ENGLISH_FIXTURE)
        segs = segment_entries(entries, language="en")
        # 60 entries spanning ~23s with multiple boundary triggers.
        assert len(segs) >= 3, f"expected ≥3 segments, got {len(segs)}"

    def test_segment_invariants(self) -> None:
        entries = _load_entries(ENGLISH_FIXTURE)
        segs = segment_entries(entries, language="en")
        for s in segs:
            assert s.start <= s.end, s
            assert s.text.strip(), f"empty text: {s}"
            assert s.words, f"english mode but no words: {s}"
            for w in s.words:
                assert w.start <= w.end, w
                assert s.start <= w.start, (s, w)
                # End may equal segment.end at boundary; allow small rounding slack
                assert w.end <= s.end + 1e-6, (s, w)

    def test_first_segment_text_sanity(self) -> None:
        entries = _load_entries(ENGLISH_FIXTURE)
        segs = segment_entries(entries, language="en")
        # First fixture entry texts: " Since" " last" " year," " I've" ...
        # The first segment's text should start with "Since" (leading space stripped).
        assert segs[0].text.startswith("Since"), segs[0].text

    def test_word_offsets_match_entries(self) -> None:
        entries = _load_entries(ENGLISH_FIXTURE)
        segs = segment_entries(entries, language="en")
        # Each segment's words inherit the entry timestamps verbatim. We
        # don't assert strict monotonicity across segments because whisper.cpp
        # DTW occasionally produces tiny back-overlaps between adjacent
        # entries (e.g. entry N ends at 9610, entry N+1 starts at 9600).
        # That's whisper-cpp's own noise, not our segmenter's bug.
        for s in segs:
            for w in s.words:
                assert s.start <= w.start + 1e-6, (s, w)
                assert w.end <= s.end + 1e-6, (s, w)

    # ------------------------------------------------------------------
    # Boundary triggers (synthetic mini cases)
    # ------------------------------------------------------------------

    def test_boundary_punct_end(self) -> None:
        # 3 entries; entry[1] ends with '.' → flush after entry[1].
        entries = [
            _mk(" Hello", 0, 200),
            _mk(" world.", 200, 500),
            _mk(" Then", 500, 700),
        ]
        segs = segment_entries(entries, language="en")
        assert len(segs) == 2
        assert segs[0].text == "Hello world."
        assert len(segs[0].words) == 2
        assert segs[1].text == "Then"
        assert len(segs[1].words) == 1

    def test_boundary_punct_with_trailing_quote(self) -> None:
        # Quote/whitespace after punctuation should still trigger.
        entries = [
            _mk(' "Hello', 0, 200),
            _mk(' world."', 200, 500),
            _mk(" Then", 500, 700),
        ]
        segs = segment_entries(entries, language="en")
        assert len(segs) == 2
        assert PUNCT_END_RE.search(' world."')

    def test_boundary_gap(self) -> None:
        # entry[1].t_from - entry[0].t_to > GAP_MS → flush after entry[0].
        entries = [
            _mk(" Hello", 0, 100),
            _mk(" world", 100 + GAP_MS + 1, 1000),
            _mk(" again", 1000, 1200),
        ]
        segs = segment_entries(entries, language="en")
        assert len(segs) == 2
        assert segs[0].text == "Hello"
        assert segs[1].text == "world again"

    def test_boundary_duration(self) -> None:
        # 11 entries each 600ms long, no gap, no punctuation. Cumulative
        # duration crosses 5000ms by entry index 8-9 → flush mid-stream.
        entries = []
        for i in range(11):
            entries.append(_mk(f" w{i}", i * 600, (i + 1) * 600))
        segs = segment_entries(entries, language="en")
        assert len(segs) >= 2, f"duration cap should trigger, got {len(segs)} segs"
        # Each segment must be within MAX_DUR_MS + one entry overshoot
        # (boundary checks AFTER append).
        for s in segs:
            dur_ms = (s.end - s.start) * 1000.0
            # Allow one-entry overshoot since boundary triggers post-append.
            assert dur_ms <= MAX_DUR_MS + 600 + 1, (s, dur_ms)

    def test_boundary_char_cap(self) -> None:
        # Build entries summing >100 chars with no punctuation, no gaps,
        # short durations → only char cap fires.
        # Each " abcdefghij" entry contributes 10 chars after strip (plus
        # one separating space when joined). Cumulative stripped length
        # crosses 100 at entry index 9 (len=109). We add 2 more so the
        # flush is clearly triggered by char cap, not EOF.
        entries = [
            _mk(" abcdefghij", i * 50, (i + 1) * 50) for i in range(12)
        ]
        segs = segment_entries(entries, language="en")
        assert len(segs) >= 2, f"char cap should trigger, got {len(segs)} segs"
        # First segment should hold the entries that pushed us over the cap.
        assert len(segs[0].words) >= 9, segs[0]

    def test_single_entry(self) -> None:
        segs = segment_entries([_mk(" lonely", 0, 500)], language="en")
        assert len(segs) == 1
        assert segs[0].text == "lonely"
        assert len(segs[0].words) == 1
        assert segs[0].words[0].word == "lonely"


# ---------------------------------------------------------------------------
# Chinese phrase mode — fixture-based
# ---------------------------------------------------------------------------


class TestChineseMode:
    def test_fixture_one_segment_per_entry(self) -> None:
        entries = _load_entries(CHINESE_FIXTURE)
        segs = segment_entries(entries, language="zh")
        assert len(segs) == len(entries) > 0

    def test_words_empty(self) -> None:
        entries = _load_entries(CHINESE_FIXTURE)
        segs = segment_entries(entries, language="zh")
        for s in segs:
            assert s.words == [], f"chinese mode segment has words: {s}"

    def test_text_preserved_stripped(self) -> None:
        entries = _load_entries(CHINESE_FIXTURE)
        segs = segment_entries(entries, language="zh")
        for s, e in zip(segs, entries, strict=True):
            assert s.text == e.text.strip()
            assert s.start == round(e.t_from_ms / 1000, 3)
            assert s.end == round(e.t_to_ms / 1000, 3)

    def test_skips_empty_text_entries(self) -> None:
        entries = [
            _mk("有内容", 0, 1000),
            _mk("   ", 1000, 1100),  # whitespace-only — should be skipped
            _mk("还有", 1100, 2000),
        ]
        segs = segment_entries(entries, language="zh")
        assert len(segs) == 2
        assert [s.text for s in segs] == ["有内容", "还有"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_returns_empty(self) -> None:
        assert segment_entries([]) == []

    def test_mixed_codeswitching_does_not_crash(self) -> None:
        entries = [
            _mk("你好", 0, 500),
            _mk("世界", 500, 1000),
            _mk("现在", 1000, 1500),
            _mk("开始", 1500, 2000),
            _mk(" Hello", 2000, 2500),
            _mk(" world", 2500, 3000),
            _mk(" foo", 3000, 3500),
            _mk(" bar", 3500, 4000),
            _mk(" baz", 4000, 4500),
            _mk(" qux", 4500, 5000),
        ]
        # 6 / 10 leading space → english_word fallback (ratio ≥ 0.5).
        segs = segment_entries(entries, language=None)
        assert len(segs) >= 1
        # Should produce SOME non-empty segment.
        assert any(s.text for s in segs)

    def test_no_speech_prob_aggregation_english(self) -> None:
        # When all entries have a no_speech_prob, segment gets the mean.
        entries = [
            Entry(text=" foo", t_from_ms=0, t_to_ms=200, no_speech_prob=0.1),
            Entry(text=" bar.", t_from_ms=200, t_to_ms=400, no_speech_prob=0.3),
        ]
        segs = segment_entries(entries, language="en")
        assert len(segs) == 1
        assert segs[0].no_speech_prob == pytest.approx(0.2)

    def test_no_speech_prob_all_none_english(self) -> None:
        entries = [_mk(" foo", 0, 200), _mk(" bar.", 200, 400)]
        segs = segment_entries(entries, language="en")
        assert segs[0].no_speech_prob is None
        assert segs[0].avg_confidence is None

    def test_seg_ids_sequential(self) -> None:
        entries = [
            _mk(" a.", 0, 100),
            _mk(" b.", 100, 200),
            _mk(" c.", 200, 300),
        ]
        segs = segment_entries(entries, language="en")
        assert [s.id for s in segs] == ["seg_001", "seg_002", "seg_003"]


# ---------------------------------------------------------------------------
# Sanity check vs the prototype dry-run
# ---------------------------------------------------------------------------


class TestPrototypeSanity:
    @pytest.mark.skipif(
        not DRYRUN_RAW.exists(),
        reason="dry-run whisper-raw.json not present",
    )
    def test_full_dryrun_segment_count_in_range(self) -> None:
        entries = _load_entries(DRYRUN_RAW)
        segs = segment_entries(entries, language="en")
        # Prototype produced 904 segments. Allow [800, 1200] window for
        # implementation drift / regex extension to \W*.
        assert 800 <= len(segs) <= 1200, (
            f"unexpected segment count {len(segs)} for {len(entries)} entries; "
            "prototype produced 904"
        )

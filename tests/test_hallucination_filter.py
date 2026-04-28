"""Tests for ``voxkit.core.hallucination_filter``."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from voxkit.core.hallucination_filter import (
    DEFAULT_BLOCKLIST_PATH,
    Blocklist,
    DroppedEntry,
    filter_entries,
    has_ghost_cjk_loop,
    load_blocklist,
    normalize,
    write_drop_log,
)
from voxkit.core.types import Entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(text: str, t_from: int = 0, t_to: int = 100) -> Entry:
    return Entry(text=text, t_from_ms=t_from, t_to_ms=t_to)


@pytest.fixture
def bl() -> Blocklist:
    return load_blocklist()


# ---------------------------------------------------------------------------
# Blocklist loading
# ---------------------------------------------------------------------------


def test_load_blocklist_default_returns_populated_blocklist() -> None:
    bl = load_blocklist()
    assert isinstance(bl, Blocklist)
    assert len(bl.watermark_prefixes) > 0
    assert len(bl.standalone_matches) > 0
    assert bl.ghost_min_chars == 6
    assert bl.ghost_min_repeats == 2
    assert bl.strip_chars  # non-empty


def test_load_blocklist_custom_path(tmp_path: Path) -> None:
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "version": 1,
                "watermark_prefixes": ["FOOBAR"],
                "standalone_matches": ["BAZ"],
                "ghost_loop": {"min_substring_chars": 4, "min_repeats": 3},
                "normalize": {"strip_chars": " ."},
            }
        ),
        encoding="utf-8",
    )
    bl = load_blocklist(custom)
    assert bl.watermark_prefixes == ("FOOBAR",)
    assert bl.standalone_matches == frozenset({"BAZ"})
    assert bl.ghost_min_chars == 4
    assert bl.ghost_min_repeats == 3
    assert bl.strip_chars == " ."


def test_load_blocklist_contains_known_remixr_entries(bl: Blocklist) -> None:
    # These are verbatim ports from the Remixr TS source — drift here means
    # the JSON has been edited away from upstream.
    assert "明镜" in bl.standalone_matches
    assert "谢谢观看" in bl.standalone_matches
    assert "字幕组" in bl.standalone_matches
    assert "优优独播剧场" in bl.watermark_prefixes
    assert "新唐人电视台" in bl.watermark_prefixes


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace() -> None:
    assert normalize(" hello ", "「」") == "hello"


def test_normalize_strips_punctuation_both_ends() -> None:
    strip = ".,。，、！？!?…-—　」「"
    assert normalize("。明镜，", strip) == "明镜"


def test_normalize_nfc_precomposed_unchanged() -> None:
    # é as a single precomposed code point (U+00E9).
    precomposed = "é"
    assert normalize(precomposed, "") == precomposed


def test_normalize_nfc_recomposes_decomposed() -> None:
    # é as e + combining acute (U+0065 U+0301) — should normalize to U+00E9.
    decomposed = "é"
    assert normalize(decomposed, "") == "é"
    # And the result is a single code point.
    assert len(normalize(decomposed, "")) == 1


def test_normalize_empty() -> None:
    assert normalize("", ".,。，") == ""


def test_normalize_whitespace_only() -> None:
    assert normalize("   ", ".,。，") == ""


def test_normalize_full_width_space() -> None:
    # U+3000 is whitespace under str.strip(); should drop.
    assert normalize("　明镜　", "") == "明镜"


# ---------------------------------------------------------------------------
# Ghost loop detection
# ---------------------------------------------------------------------------


def test_ghost_loop_detects_classic_pattern() -> None:
    text = "比较能够看起来更多的情况,比较能够看起来"
    hit = has_ghost_cjk_loop(text, 6, 2)
    assert hit is not None
    assert len(hit) == 6
    # The detected window must actually appear ≥ 2 times in the text.
    assert text.count(hit) >= 2


def test_ghost_loop_ignores_non_cjk() -> None:
    assert has_ghost_cjk_loop("hello world hello world", 6, 2) is None


def test_ghost_loop_no_repeat() -> None:
    assert has_ghost_cjk_loop("普通中文不重复内容", 6, 2) is None


def test_ghost_loop_too_short() -> None:
    assert has_ghost_cjk_loop("我", 6, 2) is None
    assert has_ghost_cjk_loop("你好世界", 6, 2) is None


def test_ghost_loop_bilingual_separator() -> None:
    text = "ABC比较能够看起来DEF比较能够看起来"
    hit = has_ghost_cjk_loop(text, 6, 2)
    assert hit is not None
    assert hit == "比较能够看起"


def test_ghost_loop_min_repeats_three() -> None:
    text = "测试一下啊好测试一下啊好测试一下啊好"
    # 3 repeats of "测试一下啊好"
    assert has_ghost_cjk_loop(text, 6, 3) is not None
    # And not enough at threshold 4.
    assert has_ghost_cjk_loop(text, 6, 4) is None


# ---------------------------------------------------------------------------
# Filter pipeline
# ---------------------------------------------------------------------------


def test_filter_drops_empty_text(bl: Blocklist) -> None:
    kept, drops = filter_entries([_entry("")], bl)
    assert kept == []
    assert len(drops) == 1
    assert drops[0].rule == "empty_after_normalize"
    assert drops[0].matched_pattern is None


def test_filter_drops_whitespace_only(bl: Blocklist) -> None:
    kept, drops = filter_entries([_entry("   ")], bl)
    assert kept == []
    assert drops[0].rule == "empty_after_normalize"


def test_filter_drops_watermark_prefix(bl: Blocklist) -> None:
    text = "优优独播剧场——YoYo Television Series Exclusive"
    kept, drops = filter_entries([_entry(text)], bl)
    assert kept == []
    assert len(drops) == 1
    assert drops[0].rule == "watermark_prefix"
    assert drops[0].matched_pattern == "优优独播剧场"
    assert drops[0].text == text


def test_filter_drops_standalone_match(bl: Blocklist) -> None:
    kept, drops = filter_entries([_entry("。明镜，")], bl)
    assert kept == []
    assert len(drops) == 1
    assert drops[0].rule == "standalone_match"
    assert drops[0].matched_pattern == "明镜"


def test_filter_drops_ghost_loop(bl: Blocklist) -> None:
    text = "比较能够看起来更多的情况,比较能够看起来更多的东西"
    kept, drops = filter_entries([_entry(text)], bl)
    assert kept == []
    assert len(drops) == 1
    assert drops[0].rule == "ghost_cjk_loop"
    assert drops[0].matched_pattern is not None
    assert len(drops[0].matched_pattern) == 6


def test_filter_keeps_clean_english(bl: Blocklist) -> None:
    kept, drops = filter_entries([_entry(" Hello world.")], bl)
    assert len(kept) == 1
    assert drops == []


def test_filter_keeps_clean_chinese(bl: Blocklist) -> None:
    kept, drops = filter_entries([_entry("你好世界")], bl)
    assert len(kept) == 1
    assert drops == []


def test_filter_keeps_substring_of_standalone(bl: Blocklist) -> None:
    # "我看着这面明镜" must NOT be dropped — "明镜" only matches as the
    # *whole* normalized entry, not as a substring.
    kept, drops = filter_entries([_entry("我看着这面明镜")], bl)
    assert len(kept) == 1
    assert drops == []


def test_filter_mixed_entries_chunk_and_indices(bl: Blocklist) -> None:
    entries = [
        _entry("Hello"),               # 0 — keep
        _entry("明镜"),                  # 1 — drop standalone
        _entry("正常的中文一句话"),         # 2 — keep
        _entry("优优独播剧场 ABC"),       # 3 — drop watermark
        _entry("正常英文 sentence"),      # 4 — keep
    ]
    kept, drops = filter_entries(entries, bl, chunk_index=7)
    assert len(kept) == 3
    assert {e.text for e in kept} == {
        "Hello",
        "正常的中文一句话",
        "正常英文 sentence",
    }
    assert len(drops) == 2
    rules = {(d.entry_index, d.rule) for d in drops}
    assert rules == {(1, "standalone_match"), (3, "watermark_prefix")}
    for d in drops:
        assert d.chunk_index == 7


# ---------------------------------------------------------------------------
# Drop log NDJSON
# ---------------------------------------------------------------------------


def test_write_drop_log_ndjson(tmp_path: Path) -> None:
    drops = [
        DroppedEntry(
            chunk_index=0,
            entry_index=3,
            text="明镜",
            rule="standalone_match",
            matched_pattern="明镜",
        ),
        DroppedEntry(
            chunk_index=0,
            entry_index=7,
            text="优优独播剧场——",
            rule="watermark_prefix",
            matched_pattern="优优独播剧场",
        ),
    ]
    log = tmp_path / "h.log"
    write_drop_log(drops, log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0 == {
        "chunk": 0,
        "entryIdx": 3,
        "text": "明镜",
        "rule": "standalone_match",
        "matchedPattern": "明镜",
    }
    row1 = json.loads(lines[1])
    assert row1["entryIdx"] == 7
    assert row1["matchedPattern"] == "优优独播剧场"


def test_write_drop_log_appends(tmp_path: Path) -> None:
    log = tmp_path / "h.log"
    drop_a = DroppedEntry(
        chunk_index=0, entry_index=0, text="a", rule="standalone_match",
        matched_pattern="a",
    )
    drop_b = DroppedEntry(
        chunk_index=1, entry_index=0, text="b", rule="watermark_prefix",
        matched_pattern="b",
    )
    write_drop_log([drop_a], log)
    write_drop_log([drop_b], log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "a"
    assert json.loads(lines[1])["text"] == "b"


# ---------------------------------------------------------------------------
# Real-world fixture sanity check
# ---------------------------------------------------------------------------


# Try a couple of CutFlow transcript fixtures; skip cleanly if they're not
# present (the package shouldn't depend on a sibling repo at runtime).
_REMIXR_CANDIDATES = [
    Path(
        "/Users/xsharp/Workspace/3Craft/CutFlow/storage/projects"
        "/proj_V-vP-yqaXHm-/sources/src_SaiCOLeO/transcript.raw.json"
    ),
    Path(
        "/Users/xsharp/Workspace/3Craft/CutFlow/storage/projects"
        "/proj_Yyrbz6bWyROG/sources/src_wfMWPDTn/transcript.raw.json"
    ),
]


def test_filter_does_not_false_positive_on_real_remixr_transcripts(
    bl: Blocklist,
) -> None:
    fixture = next((p for p in _REMIXR_CANDIDATES if p.exists()), None)
    if fixture is None:
        pytest.skip("No Remixr transcript fixture available locally.")
    data = json.loads(fixture.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    # Take the first 60 *non-empty* segments — enough to exercise the filter
    # but bounded so the test stays fast.
    sample = [s.get("text", "") for s in segments if s.get("text")][:60]
    entries = [_entry(t) for t in sample]
    kept, drops = filter_entries(entries, bl)
    # The clean Remixr corpus may contain one or two segments that the filter
    # legitimately catches as hallucinations (e.g. "中文字幕志愿者"); allow a
    # small tail but require the overwhelming majority to pass.
    assert len(kept) >= int(len(sample) * 0.85), (
        f"Too many segments dropped from real corpus: kept={len(kept)} "
        f"drops={len(drops)} of {len(sample)}; "
        f"drop rules={[d.rule for d in drops]}"
    )


def test_default_blocklist_path_is_under_voxkit_data() -> None:
    # Sanity: the bundled path actually points at our data dir.
    assert DEFAULT_BLOCKLIST_PATH.exists()
    assert DEFAULT_BLOCKLIST_PATH.parent.name == "data"
    assert DEFAULT_BLOCKLIST_PATH.parent.parent.name == "voxkit"


def test_blocklist_strip_normalization_round_trip(bl: Blocklist) -> None:
    # Re-normalizing any standalone match with the same strip set is a no-op.
    for s in bl.standalone_matches:
        assert normalize(s, bl.strip_chars) == s
        # NFC-stable too.
        assert unicodedata.normalize("NFC", s) == s

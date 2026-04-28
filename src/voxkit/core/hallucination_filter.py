"""Whisper.cpp hallucination filter.

Drops three classes of bad entries from whisper.cpp transcription output:

1. **Channel watermark prefixes** — long, distinctive YouTube channel watermark
   phrases (e.g. ``"优优独播剧场"``). Whisper hallucinates these on silent /
   non-speech segments because the training corpus was scraped from YouTube.
   Match rule: normalized entry text *starts with* the prefix (tolerates
   bilingual suffixes like ``"——YoYo Television Series Exclusive"``).

2. **Standalone hallucination phrases** — shorter phrases that *could*
   legitimately appear inside a sentence (e.g. ``"明镜"``, ``"谢谢观看"``).
   Match rule: normalized entry text is *exactly equal to* the phrase. Avoids
   false-positives on legitimate prose that happens to contain the same chars.

3. **Ghost transcript loops** — structural pattern: a CJK substring of length
   ≥ ``min_substring_chars`` that repeats ≥ ``min_repeats`` times within the
   same entry. Catches the openai/whisper Discussion #679-style failure mode
   where a weak-signal segment is misrecognized and the model self-loops on a
   high-frequency training fragment. No keyword list — pure structural check.

Ported from Remixr CutFlow ``services/whisper-hallucination-blocklist.ts``;
data lives in ``voxkit/data/hallucination_blocklist.json``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voxkit.core.types import Entry

# Default blocklist location (bundled with package).
DEFAULT_BLOCKLIST_PATH = (
    Path(__file__).parent.parent / "data" / "hallucination_blocklist.json"
)


@dataclass(frozen=True)
class Blocklist:
    """In-memory blocklist with O(1) standalone lookup."""

    watermark_prefixes: tuple[str, ...]
    standalone_matches: frozenset[str]
    ghost_min_chars: int
    ghost_min_repeats: int
    strip_chars: str


DroppedReason = Literal[
    "watermark_prefix",
    "standalone_match",
    "ghost_cjk_loop",
    "empty_after_normalize",
]


@dataclass(frozen=True)
class DroppedEntry:
    """One row of the drop log."""

    chunk_index: int
    entry_index: int
    text: str
    rule: DroppedReason
    matched_pattern: str | None


# CJK character ranges (BMP CJK Unified Ideographs + Extension A).
# Mirrors the TS implementation: U+4E00–U+9FFF and U+3400–U+4DBF.
_CJK_CHAR_RE = re.compile(r"[一-鿿㐀-䶿]")


def load_blocklist(path: Path | None = None) -> Blocklist:
    """Load blocklist JSON. ``None`` → bundled default path."""
    if path is None:
        path = DEFAULT_BLOCKLIST_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    strip_chars: str = data["normalize"]["strip_chars"]
    # Normalize the standalone-match list with the *same* strip rule we apply
    # at filter time so equality holds even if the JSON entries carry stray
    # punctuation.
    standalones = frozenset(
        normalize(s, strip_chars) for s in data["standalone_matches"]
    )
    prefixes = tuple(
        normalize(p, strip_chars) for p in data["watermark_prefixes"]
    )
    return Blocklist(
        watermark_prefixes=prefixes,
        standalone_matches=standalones,
        ghost_min_chars=int(data["ghost_loop"]["min_substring_chars"]),
        ghost_min_repeats=int(data["ghost_loop"]["min_repeats"]),
        strip_chars=strip_chars,
    )


def normalize(text: str, strip_chars: str) -> str:
    """NFC-normalize, then strip whitespace + strip the configured punctuation
    chars from both ends.

    Order matters:
      1. NFC compose any decomposed CJK / accented chars.
      2. ``str.strip()`` — drops Unicode whitespace (incl. full-width space if
         caller didn't list it, though we typically include U+3000).
      3. ``str.strip(strip_chars)`` — drops the configured punctuation chars.
    """
    nfc = unicodedata.normalize("NFC", text)
    return nfc.strip().strip(strip_chars)


def has_ghost_cjk_loop(
    text: str, min_chars: int, min_repeats: int
) -> str | None:
    """Return the first repeated all-CJK substring of length ``min_chars`` that
    occurs ≥ ``min_repeats`` times, else ``None``.

    Algorithm: O(N · L) sliding window where L = ``min_chars`` (constant 6 in
    practice). For each window position we cheaply reject non-CJK windows by
    char-class regex match; surviving windows go into a dict counter. Returns
    on first window to hit the repeat threshold so we short-circuit on long
    self-looping entries.
    """
    if min_chars <= 0 or min_repeats <= 1:
        return None
    # Quick length floor: need at least min_chars * min_repeats characters
    # *somewhere* before any window can repeat.
    if len(text) < min_chars * min_repeats:
        return None
    seen: dict[str, int] = {}
    n = len(text)
    for i in range(n - min_chars + 1):
        window = text[i : i + min_chars]
        all_cjk = True
        for ch in window:
            if not _CJK_CHAR_RE.match(ch):
                all_cjk = False
                break
        if not all_cjk:
            continue
        next_count = seen.get(window, 0) + 1
        if next_count >= min_repeats:
            return window
        seen[window] = next_count
    return None


def filter_entries(
    entries: list[Entry],
    blocklist: Blocklist,
    *,
    chunk_index: int = 0,
) -> tuple[list[Entry], list[DroppedEntry]]:
    """Apply the four-rule pipeline per entry, in order. Pure function.

    Rules (first match wins):
      1. ``empty_after_normalize`` — normalized text is empty.
      2. ``watermark_prefix`` — normalized text starts with a configured
         prefix.
      3. ``standalone_match`` — normalized text *equals* a standalone phrase.
      4. ``ghost_cjk_loop`` — original text contains a repeated CJK substring.

    Note rule 4 uses the *original* (un-normalized) text, matching the Remixr
    TS implementation — punctuation between repeats can break the substring,
    so stripping it would actually weaken detection.
    """
    kept: list[Entry] = []
    drops: list[DroppedEntry] = []
    for idx, entry in enumerate(entries):
        norm = normalize(entry.text, blocklist.strip_chars)
        if not norm:
            drops.append(
                DroppedEntry(
                    chunk_index=chunk_index,
                    entry_index=idx,
                    text=entry.text,
                    rule="empty_after_normalize",
                    matched_pattern=None,
                )
            )
            continue
        prefix_hit: str | None = None
        for prefix in blocklist.watermark_prefixes:
            if prefix and norm.startswith(prefix):
                prefix_hit = prefix
                break
        if prefix_hit is not None:
            drops.append(
                DroppedEntry(
                    chunk_index=chunk_index,
                    entry_index=idx,
                    text=entry.text,
                    rule="watermark_prefix",
                    matched_pattern=prefix_hit,
                )
            )
            continue
        if norm in blocklist.standalone_matches:
            drops.append(
                DroppedEntry(
                    chunk_index=chunk_index,
                    entry_index=idx,
                    text=entry.text,
                    rule="standalone_match",
                    matched_pattern=norm,
                )
            )
            continue
        ghost = has_ghost_cjk_loop(
            entry.text, blocklist.ghost_min_chars, blocklist.ghost_min_repeats
        )
        if ghost is not None:
            drops.append(
                DroppedEntry(
                    chunk_index=chunk_index,
                    entry_index=idx,
                    text=entry.text,
                    rule="ghost_cjk_loop",
                    matched_pattern=ghost,
                )
            )
            continue
        kept.append(entry)
    return kept, drops


def write_drop_log(drops: list[DroppedEntry], path: Path) -> None:
    """Append drops to ``path`` as NDJSON (one entry per line).

    Append mode is intentional: a workspace can process multiple chunks across
    invocations, and each ``filter_entries`` call should add to the existing
    log rather than truncate it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for d in drops:
            row = {
                "chunk": d.chunk_index,
                "entryIdx": d.entry_index,
                "text": d.text,
                "rule": d.rule,
                "matchedPattern": d.matched_pattern,
            }
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


__all__ = [
    "DEFAULT_BLOCKLIST_PATH",
    "Blocklist",
    "DroppedEntry",
    "DroppedReason",
    "filter_entries",
    "has_ghost_cjk_loop",
    "load_blocklist",
    "normalize",
    "write_drop_log",
]

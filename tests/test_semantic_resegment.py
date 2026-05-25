"""Unit tests for voxkit.core.semantic_resegment.

纯函数测试：构造 RemixrSegment[] 输入，验证三条分支：

  1. CJK pass-through：每段直接 1:1 映射成 cue，不调 pysbd
  2. 无 word 时间戳的非 CJK：同样 pass-through（防御性）
  3. 英文有 word 时间戳：调 pysbd，按句子边界切分；超长句调 split_long
"""

from __future__ import annotations

import pytest

from voxkit.core.semantic_resegment import (
    ResegmentParams,
    SubtitleCue,
    resegment_for_subtitles,
)
from voxkit.io.schema import RemixrSegment, RemixrWord


def _seg(
    sid: str,
    start: float,
    end: float,
    text: str,
    words: list[RemixrWord] | None = None,
    speaker: str = "Speaker 1",
) -> RemixrSegment:
    return RemixrSegment(
        id=sid,
        speaker=speaker,
        start=start,
        end=end,
        text=text,
        words=words or [],
    )


def _w(word: str, start: float, end: float) -> RemixrWord:
    return RemixrWord(word=word, start=start, end=end)


# ─────────────────────────────────────────────────────────────────────────
# Pass-through branches
# ─────────────────────────────────────────────────────────────────────────


def test_empty_input_returns_empty():
    assert resegment_for_subtitles([], language="en") == []


def test_cjk_long_cues_passthrough_unchanged():
    """CJK 路径不调 pysbd；当 cue 已经 ≥ min_dur_s 时不该合并，1:1 输出。"""
    segs = [
        _seg("s1", 0.0, 2.0, "你好世界。", speaker="Speaker 1"),
        _seg("s2", 2.0, 4.0, "今天天气真好。", speaker="Speaker 1"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    assert len(cues) == 2
    assert cues[0].text == "你好世界。"
    assert cues[0].speaker == "Speaker 1"
    assert cues[1].text == "今天天气真好。"


def test_cjk_short_cues_get_merged():
    """档 1：CJK 路径打开 _merge_too_short——同 speaker 短 cue 合并到邻居，
    避免 < 1.5s 的闪现字幕。
    """
    # 三个 1s 同 speaker cue → 单调合并到 max_dur_s 上限附近
    segs = [
        _seg("s1", 0.0, 1.0, "第一句", speaker="Speaker 1"),
        _seg("s2", 1.0, 2.0, "第二句", speaker="Speaker 1"),
        _seg("s3", 2.0, 3.0, "第三句", speaker="Speaker 1"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    # 至少减少 cue 数；且合并后每个 cue ≥ min_dur_s（除非物理上限阻挡）
    assert len(cues) < 3
    # 合并后的文本保留单空格分隔
    joined = " ".join(c.text for c in cues)
    assert "第一句" in joined and "第二句" in joined and "第三句" in joined


def test_cjk_short_cues_different_speakers_dont_merge():
    """跨 speaker 不合并——即使两侧都很短，也保留为独立 cue（避免说话人混淆）。"""
    segs = [
        _seg("s1", 0.0, 1.0, "你好", speaker="Speaker 1"),
        _seg("s2", 1.0, 2.0, "Hi", speaker="Speaker 2"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    assert len(cues) == 2
    assert cues[0].speaker == "Speaker 1"
    assert cues[1].speaker == "Speaker 2"


def test_passthrough_when_words_missing():
    """非 CJK 但 segments[0].words 为空 → 走 CJK-style 路径（passthrough + 短合并）。"""
    segs = [_seg("s1", 0.0, 2.0, "Hello world.", words=[])]
    cues = resegment_for_subtitles(segs, language="en")
    assert len(cues) == 1
    assert cues[0].text == "Hello world."


def test_passthrough_skips_blank_text():
    """空白 segment 在入口就被滤掉，不参与合并候选。"""
    segs = [
        # 两侧 cue 时长 ≥ min_dur_s，确保不会合并；专测空白过滤
        _seg("s1", 0.0, 2.0, "Hello.", speaker="A"),
        _seg("s2", 2.0, 3.0, "   ", speaker="A"),
        _seg("s3", 3.0, 5.0, "World.", speaker="A"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    assert [c.text for c in cues] == ["Hello.", "World."]


# ─────────────────────────────────────────────────────────────────────────
# English word-level path
# ─────────────────────────────────────────────────────────────────────────


def test_simple_two_sentence_segment_splits_at_period():
    """同一 segment 含两句话 → 按句号切成 2 cue。"""
    segs = [
        _seg(
            "s1",
            0.0,
            4.0,
            "Hello world. This is fine.",
            words=[
                _w("Hello", 0.0, 0.4),
                _w("world.", 0.4, 0.9),
                _w("This", 1.0, 1.3),
                _w("is", 1.3, 1.5),
                _w("fine.", 1.5, 2.0),
            ],
            speaker="Speaker 1",
        ),
    ]
    # min_dur_s 默认 1.5，会触发短 cue 合并；用更宽松参数关掉合并验证 SBD 单独效果
    p = ResegmentParams(min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    assert len(cues) == 2
    assert cues[0].text.endswith(".")
    assert cues[1].text.endswith(".")
    assert cues[0].text.startswith("Hello")
    assert cues[1].text.startswith("This")


def test_speaker_boundary_blocks_pysbd_merge():
    """跨 speaker 的"句子"不能被 pysbd 当成一句合并。"""
    segs = [
        _seg(
            "s1",
            0.0,
            1.0,
            "Hello",
            words=[_w("Hello", 0.0, 0.4)],
            speaker="Speaker 1",
        ),
        _seg(
            "s2",
            1.0,
            2.0,
            "world.",
            words=[_w("world.", 1.0, 1.4)],
            speaker="Speaker 2",
        ),
    ]
    p = ResegmentParams(min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    speakers = {c.speaker for c in cues}
    assert speakers == {"Speaker 1", "Speaker 2"}, "speakers must not be merged"


def test_long_sentence_gets_split():
    """单句 >max_chars + max_dur → split_long 切多段，每段都合规。"""
    # 构造一句 90 词、~30s 的句子，没有句号但有逗号
    pieces: list[tuple[str, float]] = []
    t = 0.0
    for i in range(60):
        pieces.append((f"word{i},", t))
        t += 0.4
    pieces.append(("end.", t))
    words = [_w(w, s, s + 0.3) for w, s in pieces]
    text = " ".join(w for w, _ in pieces)
    segs = [_seg("s1", 0.0, t + 0.3, text, words=words, speaker="A")]

    p = ResegmentParams(max_dur_s=6.0, max_chars=70, min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)

    assert len(cues) >= 3, "long sentence must be split"
    for c in cues:
        assert c.end - c.start <= p.max_dur_s * 1.21, f"cue {c} duration over hard limit"
        assert len(c.text) <= p.max_chars * 1.31, f"cue {c} chars over hard limit"


def test_short_cue_merged_with_neighbour():
    """<min_dur_s 的小尾巴应被合并到同 speaker 邻居。"""
    segs = [
        _seg(
            "s1",
            0.0,
            5.0,
            "First long sentence here. Tiny.",
            words=[
                _w("First", 0.0, 0.3),
                _w("long", 0.3, 0.6),
                _w("sentence", 0.6, 1.0),
                _w("here.", 1.0, 1.4),
                _w("Tiny.", 1.5, 2.0),  # 0.5s — well below min_dur=1.5
            ],
            speaker="A",
        ),
    ]
    p = ResegmentParams(min_dur_s=1.5)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    # "Tiny." 应该被合进前一个 cue
    assert all(c.end - c.start >= 0.05 for c in cues)
    # 至少一个 cue 同时含 "here." 和 "Tiny."（合并后）
    merged = [c for c in cues if "here." in c.text and "Tiny." in c.text]
    assert merged, f"short cue not merged into neighbour; got: {[c.text for c in cues]}"


def test_monotonic_timestamps_enforced():
    """whisper word ts 偶尔倒挂（next.start < prev.end）→ 输出必须单调。"""
    segs = [
        _seg(
            "s1",
            0.0,
            3.0,
            "Sentence one. Sentence two.",
            words=[
                _w("Sentence", 0.0, 0.5),
                _w("one.", 0.5, 1.0),
                # 故意倒挂：next.start 在 prev.end 之前
                _w("Sentence", 0.95, 1.5),
                _w("two.", 1.5, 2.0),
            ],
            speaker="A",
        ),
    ]
    p = ResegmentParams(min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    # 后一 cue 的 start 必 >= 前一 cue 的 end
    for i in range(len(cues) - 1):
        assert cues[i + 1].start >= cues[i].end


# ─────────────────────────────────────────────────────────────────────────
# v0.5.1 切分质量回归（基于 e2e_test 真实 case）
# ─────────────────────────────────────────────────────────────────────────


def test_split_long_avoids_preposition_trailing():
    """长句被 split_long 切分时，应避开介词/连词收尾的切点。

    复现 e2e cue_000001-002 的 "...he's certainly got some / scars around..."
    问题：原算法因 prosody gap 偏好把切点选在 "some" 后；新算法对 trailing-bad
    token 减权，应选别处。
    """
    # 构造无句号的长句，迫使 split_long 走软切点路径
    # "we built some great products and shipped them globally for our customers and partners"
    # 95 chars, 5.6s, 14 words —— 必须 split (max_chars=70)
    pieces = [
        ("we", 0.0, 0.2),
        ("built", 0.2, 0.5),
        ("some", 0.5, 0.7),     # trailing-bad
        ("great", 0.8, 1.1),    # 0.1s gap → prosody gap 让 "some" 后变诱人
        ("products", 1.1, 1.6),
        ("and", 1.6, 1.7),      # trailing-bad
        ("shipped", 1.8, 2.2),
        ("them", 2.2, 2.4),
        ("globally", 2.4, 2.9),
        ("for", 2.9, 3.0),      # trailing-bad
        ("our", 3.1, 3.3),      # trailing-bad
        ("customers", 3.3, 4.0),
        ("and", 4.0, 4.1),      # trailing-bad
        ("partners", 4.1, 4.7),
    ]
    words = [_w(w, s, e) for w, s, e in pieces]
    text = " ".join(w for w, _, _ in pieces)
    segs = [_seg("s1", 0.0, 4.7, text, words=words, speaker="A")]
    p = ResegmentParams(max_chars=50, max_dur_s=10.0, min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)

    assert len(cues) >= 2, f"long line must split; got {len(cues)} cues"
    bad_endings = {"some", "and", "for", "our", "the", "of"}
    for c in cues[:-1]:  # 末尾 cue 没有"下一个"，不计
        last_token = c.text.rstrip().split()[-1].rstrip(".,!?;:").lower()
        assert last_token not in bad_endings, (
            f"cue ends in trailing-bad token {last_token!r}: {c.text!r}; "
            f"all cues={[x.text for x in cues]}"
        )


def test_merge_too_short_handles_flash_cue_with_tight_cps():
    """复现 e2e cue_000008 'I'll' 0.17s 闪屏 case。

    原算法：合并后 cps 22.4 > max_cps 22 → 拒合并 → 0.17s 闪屏字幕保留。
    新算法：< 0.5s flash cue 触发 cps/chars 放宽 1.5×/1.2×，必须吸入邻居。
    """
    # prev: 5.92s "long sentence one"，next: 2.46s "I'll be back at the end"
    # 中间夹一个 0.17s "I'll" 闪屏；合并后 prev+flash = 6.09s, ~60 chars
    # 原 max_cps=22 边缘卡住；放宽后 33 cps 容忍
    segs = [
        _seg(
            "s1",
            0.0,
            8.55,
            ("and he had some really interesting thoughts on the job of a "
             "modern ceo So tune in. I'll be back at the end and give you "
             "some further thoughts."),
            words=(
                # 第一段：长句 ~5.92s 收在 "in."
                [_w("and", 0.0, 0.1), _w("he", 0.1, 0.2), _w("had", 0.2, 0.4),
                 _w("really", 0.4, 0.7), _w("interesting", 0.7, 1.2),
                 _w("thoughts.", 1.2, 1.6), _w("So", 4.0, 4.4),
                 _w("tune", 4.4, 5.5), _w("in.", 5.5, 5.92)]
                # 闪屏 word
                + [_w("I'll", 5.92, 6.09)]
                # 后续句子
                + [_w("be", 6.09, 6.3), _w("back.", 6.3, 6.7),
                   _w("End", 7.0, 7.5), _w("of", 7.5, 7.7),
                   _w("thoughts.", 7.7, 8.55)]
            ),
            speaker="A",
        ),
    ]
    p = ResegmentParams(min_dur_s=1.5)
    cues = resegment_for_subtitles(segs, language="en", params=p)
    # 关键断言：没有任何 cue 时长 < 0.5s（flash 应被合并掉）
    flashes = [c for c in cues if (c.end - c.start) < 0.5]
    assert not flashes, (
        f"flash cue should be merged into neighbour; got: "
        f"{[(c.text, c.end - c.start) for c in flashes]}"
    )


def test_split_long_prefers_clause_punctuation_over_prosody_gap():
    """逗号 / 句末标点切点优先级保持高于 prosody gap（不被新词性折扣误伤）。"""
    pieces = [
        ("First", 0.0, 0.4),
        ("clause,", 0.4, 0.8),  # 逗号 - 强切点
        # 长 prosody gap 在介词后（应被压权），但同行有逗号 → 应优先选逗号
        ("after", 1.5, 1.9),    # gap 0.7s
        ("the", 1.9, 2.0),      # trailing-bad
        ("comma", 3.0, 3.4),    # gap 1.0s + 介词收尾 → 应被压
        ("we", 3.4, 3.6),
        ("continue.", 3.6, 4.2),
    ]
    words = [_w(w, s, e) for w, s, e in pieces]
    text = " ".join(w for w, _, _ in pieces)
    segs = [_seg("s1", 0.0, 4.2, text, words=words, speaker="A")]
    p = ResegmentParams(max_chars=20, max_dur_s=10.0, min_dur_s=0.0)
    cues = resegment_for_subtitles(segs, language="en", params=p)

    # 至少有一个切点；且第一个 cue 应该停在逗号后（"First clause,"）
    assert len(cues) >= 2
    assert cues[0].text.endswith(","), (
        f"first cue should end at comma, got: {cues[0].text!r}"
    )


def test_pathological_long_word_does_not_recurse_forever():
    """whisper-cli 偶发把长静音锁进单词的 end → 单词 dur 超 _HARD_DUR_RATIO。

    原 _split_long 在该输入下：start=0 时所有候选切点 chunk_dur 都 > max_dur_s*1.2
    被 hard-ratio 全数拒绝 → best_idx=-1 → 主循环立即 break → chunks=[words]
    （没收缩）→ 递归保护条件 `_need_split(chunk) and len(chunk)>=4` 仍为真
    → 同一输入再次递归 → RecursionError。

    新版应：检测到子 chunk 没收缩时停止递归，把该 chunk 原样输出。
    """
    # 第一个 word "Steam" 端到端 15s（病态尾随静音），其余正常
    words = [
        _w("Steam", 0.0, 15.0),
        _w("rolled", 15.0, 15.4),
        _w("over", 15.4, 15.6),
        _w("us", 15.6, 16.0),
    ]
    segs = [_seg("s1", 0.0, 16.0, "Steam rolled over us", words=words, speaker="A")]
    p = ResegmentParams(min_dur_s=0.0)

    # 关键：不应抛 RecursionError；至少产出一个 cue，文本含全部 token
    cues = resegment_for_subtitles(segs, language="en", params=p)
    assert cues, "should converge and produce at least one cue"
    joined = " ".join(c.text for c in cues)
    for token in ("Steam", "rolled", "over", "us"):
        assert token in joined, f"token {token!r} lost; cues={[c.text for c in cues]}"


def test_pysbd_missing_raises_import_error(monkeypatch):
    """pysbd 不可用时模块直接抛 ImportError，由 caller 决定 fallback。"""
    import sys
    monkeypatch.setitem(sys.modules, "pysbd", None)
    segs = [
        _seg("s1", 0.0, 1.0, "Hi.", words=[_w("Hi.", 0.0, 0.3)]),
    ]
    with pytest.raises((ImportError, TypeError)):
        # TypeError because monkeypatching to None makes import fail with TypeError on Python 3.10+
        resegment_for_subtitles(segs, language="en")

"""CJK character-level semantic resegmentation tests."""

from __future__ import annotations

from voxkit.core.semantic_resegment import ResegmentParams, resegment_for_subtitles
from voxkit.io.schema import RemixrSegment


def _seg(
    sid: str,
    start: float,
    end: float,
    text: str,
    speaker: str = "Speaker 1",
) -> RemixrSegment:
    return RemixrSegment(
        id=sid,
        speaker=speaker,
        start=start,
        end=end,
        text=text,
        words=[],
    )


def test_cjk_long_sentence_splits_on_sentence_punctuation():
    segs = [
        _seg("s1", 0.0, 6.0, "第一句说完了。第二句继续讲！第三句也结束？"),
    ]
    p = ResegmentParams(min_dur_s=0.0, max_chars=40, max_dur_s=7.0)

    cues = resegment_for_subtitles(segs, language="zh", params=p)

    assert [c.text for c in cues] == [
        "第一句说完了。",
        "第二句继续讲！",
        "第三句也结束？",
    ]
    assert cues[0].start == 0.0
    assert cues[0].end <= cues[1].start <= cues[1].end <= cues[2].start <= cues[2].end


def test_cjk_short_sentences_merge_to_avoid_flashing():
    segs = [
        _seg("s1", 0.0, 0.5, "好。"),
        _seg("s2", 0.5, 1.0, "是。"),
        _seg("s3", 1.0, 2.2, "走吧。"),
    ]
    p = ResegmentParams(min_dur_s=1.5, max_chars=20, max_dur_s=7.0)

    cues = resegment_for_subtitles(segs, language="zh", params=p)

    assert len(cues) == 1
    assert cues[0].text == "好。是。走吧。"
    assert cues[0].end - cues[0].start >= p.min_dur_s


def test_cjk_different_speakers_never_merge():
    segs = [
        _seg("s1", 0.0, 0.8, "你好。", speaker="Speaker 1"),
        _seg("s2", 0.8, 1.6, "你好。", speaker="Speaker 2"),
    ]
    p = ResegmentParams(min_dur_s=1.5, max_chars=20, max_dur_s=7.0)

    cues = resegment_for_subtitles(segs, language="zh", params=p)

    assert len(cues) == 2
    assert [c.speaker for c in cues] == ["Speaker 1", "Speaker 2"]
    assert [c.text for c in cues] == ["你好。", "你好。"]


def test_cjk_long_unpunctuated_text_splits_by_physical_limits():
    text = "这是一个没有任何标点的很长中文句子用来验证物理约束会兜底切分字幕"
    segs = [_seg("s1", 0.0, 12.0, text)]
    p = ResegmentParams(min_dur_s=0.0, max_chars=12, max_dur_s=3.0, max_cps=8.0)

    cues = resegment_for_subtitles(segs, language="zh", params=p)

    assert len(cues) >= 4
    assert "".join(c.text for c in cues) == text
    for cue in cues:
        assert len(cue.text) <= p.max_chars
        assert cue.end - cue.start <= p.max_dur_s * 1.2


def test_cjk_phrase_boundaries_are_preserved_for_real_smoke_sample():
    """Regression target from tmp/.../subtitles.proposed.srt analysis.

    The old char-first splitter cut inside phrases such as 很难很难, 这种改变,
    苏理士, and 佛罗伊德. Phrase-aware packing may still combine neighbouring
    ASR phrases, but it should not split those phrases internally.
    """
    segs = [
        _seg("s01", 0.00, 0.34, "和"),
        _seg("s02", 0.34, 0.57, "AI"),
        _seg("s03", 0.57, 2.10, "共生很难很难"),
        _seg("s04", 2.10, 3.92, "整段觉得我能够用AI就叫AI"),
        _seg("s05", 3.92, 4.24, "原生"),
        _seg("s06", 4.24, 5.02, "那就是瞎扯"),
        _seg("s07", 5.02, 5.17, "AI"),
        _seg("s08", 5.17, 7.14, "原生它是一种思维范式的一种"),
        _seg("s09", 7.14, 8.46, "根本性的这种改变"),
        _seg("s10", 8.46, 8.63, "AI"),
        _seg("s11", 8.63, 10.94, "时代即使对第一次工业革命的"),
        _seg("s12", 10.94, 13.06, "否定也是对它的翻诚"),
        _seg("s13", 13.06, 14.34, "现在是最伟大的时代"),
        _seg("s14", 14.34, 17.16, "那为什么说要把所有的行政职务全部给辞较"),
        _seg("s15", 17.16, 18.06, "道理非常简单"),
        _seg("s16", 18.06, 19.26, "不想错过这个时代"),
        _seg("s17", 19.26, 23.50, "先请您做一下简单介绍"),
        _seg("s18", 23.50, 24.02, "OK"),
        _seg("s19", 24.02, 24.64, "各位好"),
        _seg("s20", 24.64, 25.58, "我叫刘家"),
        _seg("s21", 25.58, 27.86, "我来高中作数律化竞赛"),
        _seg("s22", 27.86, 30.30, "当时对苏理士特别感兴趣"),
        _seg("s23", 30.30, 32.14, "就准备选择物理"),
        _seg("s24", 32.14, 34.66, "作为我的这一辈子的工作方向"),
        _seg("s25", 34.66, 36.46, "后来高山的小儿多了一本书"),
        _seg("s26", 36.46, 39.34, "佛罗伊德的转积文学叫新年的激情"),
        _seg("s27", 39.34, 41.98, "当时我就觉得心理世界比物理世界好玩多了"),
        _seg("s28", 41.98, 44.66, "所以我在大学的时候我就选择了学心理学"),
        _seg("s29", 44.66, 45.18, "去了"),
    ]

    cues = resegment_for_subtitles(segs, language="zh")
    texts = [c.text for c in cues]
    joined = "".join(texts)

    assert joined == "".join(s.text for s in segs)
    # Phase 2 收紧 _CJK_DEFAULT_SOFT_MAX_CHARS 28→18 后由 11 cue 变 15 cue（更细）
    assert len(cues) == 15
    # 软上限是 18，但 atom 自身长度可短暂越界到硬上限 42 以内；锁死实测 max=23
    assert max(len(t) for t in texts) <= 23
    assert all(c.end - c.start >= 1.5 for c in cues)
    # Phrase 完整性断言：以下短语不应被切到多个 cue（这是本测试的核心目的）
    assert any("共生很难很难" in t for t in texts)
    assert any("根本性的这种改变" in t for t in texts)
    assert any("当时对苏理士特别感兴趣" in t for t in texts)
    assert any("佛罗伊德的转积文学叫新年的激情" in t for t in texts)
    assert not any(t.startswith(("难", "种", "士", "罗伊德")) for t in texts)


# ── Phase 2 regressions: dense default + Latin-word atomicity ───────────────


def test_cjk_split_does_not_break_latin_words_internally():
    """长 CJK segment 含 'Steam'，物理上限触发 split 时不能切到拉丁词内部。

    Regression：voxkit 0.5.1 把 'Steam手柄是一个对称布局的' (7.85s) 切成
    '...买？S t' + 'eam 手柄...'（cue 8/9 跨边界切了 'Steam'）。
    """
    import re

    segs = [
        _seg("s1", 0.0, 4.5, "它到底值不值得买"),
        _seg("s2", 4.5, 12.3, "Steam手柄是一个对称布局的"),
    ]
    # max_dur_s 短于 s2 时长，强制走 _split_cjk_atom 路径
    p = ResegmentParams(min_dur_s=0.0, max_chars=20, max_dur_s=4.0)

    cues = resegment_for_subtitles(segs, language="zh", params=p)
    joined = "".join(c.text for c in cues)

    assert "Steam" in joined, f"Steam 被拆没了: cues={[c.text for c in cues]}"
    # 不允许任何 cue 末尾是 1-3 个孤立拉丁字母（紧邻空白 / CJK / 标点 / 行首）
    for cue in cues:
        assert not re.search(
            r"(?:^|[\s　-〿＀-￯一-鿿])[A-Za-z]{1,3}\s*$",
            cue.text,
        ), f"cue 末尾是孤立拉丁字母片段，'Steam' 类词被切断: {cue.text!r}"


def test_cjk_build_atoms_splits_on_medium_punctuation():
    """带逗号 / 顿号 / 冒号的中文 segment 在 atom 构建阶段就被切分。

    Regression：voxkit ≤0.6.0 的 `_build_cjk_atoms` 只在「。！？；」切，
    完全忽略 `_CJK_MEDIUM_BREAK = ，、：:`。当输入已带标点时（proofread
    后双 pass 场景），逗号承载 ~80% 气口边界，全部不切就会让 cue 仍被
    packing 阶段合并过粗。

    本测试用大字符上限避免 packing 阶段合并，专注验证 atom 切分。
    """
    # 单个长 segment 内含 2 个逗号 + 1 个句号
    segs = [
        _seg("s1", 0.0, 6.0, "你好，我叫张三，今天天气真好。"),
    ]
    # max_chars=5 强制 packing 阶段不合并相邻 atom（每段 ≥ 4 字符就独立 cue）
    p = ResegmentParams(min_dur_s=0.0, max_chars=5, max_dur_s=7.0)

    cues = resegment_for_subtitles(segs, language="zh", params=p)

    # 期望切成 3 条独立 cue：以逗号、逗号、句号收尾
    assert len(cues) >= 3, f"逗号未生效，cues={[c.text for c in cues]}"
    assert cues[0].text.endswith("，"), f"cue 0 末尾不是逗号: {cues[0].text!r}"
    assert cues[1].text.endswith("，"), f"cue 1 末尾不是逗号: {cues[1].text!r}"
    # 最后一条以句号结尾（原 _CJK_SENTENCE_END 路径）
    assert cues[-1].text.endswith("。"), f"末 cue 末尾不是句号: {cues[-1].text!r}"
    # 文本必须完整保留
    assert "".join(c.text for c in cues) == "你好，我叫张三，今天天气真好。"


def test_cjk_default_packs_unpunctuated_chinese_at_vlog_density():
    """Vlog 风格无标点中文，默认参数下平均 cue 字符数应贴近金标（≤ 18）。

    Regression：voxkit 0.5.1 默认把这段打包到 avg 23 char/cue（金标 11）。
    收紧 `_CJK_DEFAULT_SOFT_MAX_CHARS` 后期望 avg ≤ 18。
    """
    # 小宁子开场风格：whisper segments 已按气口切得很细
    segs = [
        _seg("s01", 0.00, 1.80, "Steam出新硬件了"),
        _seg("s02", 1.80, 2.60, "是个手柄"),
        _seg("s03", 2.60, 4.00, "卖700块"),
        _seg("s04", 4.00, 6.40, "刚出几个小时就全网断货"),
        _seg("s05", 6.40, 7.70, "我这个也是我半夜"),
        _seg("s06", 7.70, 9.30, "狂按刷新键抢到的"),
    ]
    cues = resegment_for_subtitles(segs, language="zh")
    avg = sum(len(c.text) for c in cues) / len(cues)
    assert avg <= 18, (
        f"avg chars/cue {avg:.1f} > 18; "
        f"reseg 把无标点中文打包过粗，cues={[c.text for c in cues]}"
    )

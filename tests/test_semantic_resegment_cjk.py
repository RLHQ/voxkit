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
    assert len(cues) == 11
    assert max(len(t) for t in texts) <= 28
    assert all(c.end - c.start >= 1.5 for c in cues)
    assert any("共生很难很难" in t for t in texts)
    assert any("根本性的这种改变" in t for t in texts)
    assert any("当时对苏理士特别感兴趣" in t for t in texts)
    assert any("佛罗伊德的转积文学叫新年的激情" in t for t in texts)
    assert not any(t.startswith(("难", "种", "士", "罗伊德")) for t in texts)
    assert not any(t.endswith(("很", "这", "苏理", "佛")) for t in texts)

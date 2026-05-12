"""英文不应作 cue 末尾的"不完整"词集合。

包含介词、冠词、助动词、并列连词、关系/疑问代词、不定代词、缩略 will/would 等。
词表故意保守（~150 词），不引入 spaCy；目标是把 cue 末尾停在介词/连词的比例从
40% 压到 < 15%，覆盖典型口语场景，剩余靠 LLM proofread 兜底。

CJK 主体路径不调用 :func:`is_trailing_bad`（中文/日文没有 token 概念，且 CJK
字幕的语义边界由字符插值/标点决定，不是按词性）。
"""

from __future__ import annotations

__all__ = ["ENGLISH_TRAILING_BAD", "is_trailing_bad"]


ENGLISH_TRAILING_BAD: frozenset[str] = frozenset({
    # 冠词
    "a", "an", "the",
    # 介词
    "of", "in", "on", "at", "by", "for", "with", "from", "to", "into", "onto",
    "upon", "about", "above", "below", "under", "over", "between", "through",
    "during", "across", "behind", "before", "after", "without", "within",
    "against", "around", "near", "off", "out", "up", "down",
    # 并列 / 从属连词
    "and", "or", "but", "nor", "so", "yet",
    "if", "when", "while", "because", "since", "though", "although",
    "as", "than", "that", "whether",
    # 助动词 / 系动词
    "is", "are", "was", "were", "am", "be", "being", "been",
    "do", "does", "did", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # 缩略
    "i'll", "you'll", "we'll", "they'll", "he'll", "she'll", "it'll",
    "i'm", "you're", "we're", "they're",
    "i've", "you've", "we've", "they've",
    "i'd", "you'd", "we'd", "they'd", "he'd", "she'd",
    # 代词 / 限定词（弱位置：放句末通常意味着后续有内容）
    "i", "you", "we", "they", "he", "she", "it",
    "this", "that", "these", "those",
    "my", "your", "our", "their", "his", "her", "its",
    "some", "any", "all", "each", "every", "no", "both", "many", "few",
    "much", "more", "most", "such",
    # 关系 / 疑问
    "who", "whom", "whose", "which", "what", "where", "why", "how",
    # 高频副词 / 助词（口语里常作犹豫词放句末半截后续未说）
    "not", "very", "quite", "just", "really", "kind",
})


# 任何标点收尾都豁免：句末 .!? 是真正句末；逗号/分号/冒号是说话人或 LLM
# 认可的停顿点（朗读时有自然换气），切在那里完全可接受。我们要打压的是
# **完全无标点的裸介词/裸连词收尾**（"got some" / "like the"）。
_PAUSE_PUNCT = ".!?,;:"
_STRIP = "\"'`()[]{}"


def is_trailing_bad(token: str) -> bool:
    """token 是否不该作 cue 末尾。

    任何带标点收尾的 token 直接豁免（标点 = 合法停顿点）。否则剥引号/括号 +
    小写后查集合。空 token / 仅标点 → ``False``。
    """
    bare = token.strip().strip(_STRIP)
    if not bare:
        return False
    if bare[-1] in _PAUSE_PUNCT:
        return False
    return bare.lower() in ENGLISH_TRAILING_BAD

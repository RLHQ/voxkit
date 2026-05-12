"""Prompt 加载器测试。"""

from __future__ import annotations

import re

import pytest

from voxkit.llm.errors import LLMError
from voxkit.llm.prompts import load_prompt


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_load_proofread_v1_returns_text_and_hash() -> None:
    text, digest = load_prompt("proofread")
    assert text.strip()  # 非空（placeholder 也至少有一行）
    assert _HEX64.match(digest)


def test_hash_is_stable_across_calls() -> None:
    _, d1 = load_prompt("proofread")
    _, d2 = load_prompt("proofread")
    assert d1 == d2


def test_translate_v1_loadable() -> None:
    text, digest = load_prompt("translate")
    assert text.strip()
    assert _HEX64.match(digest)


def test_unknown_stage_raises() -> None:
    with pytest.raises(LLMError):
        load_prompt("does-not-exist")

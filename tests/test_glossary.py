"""glossary 模块单测：加载、hash 稳定性、protected 集合。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from voxkit.io.glossary import (
    Glossary,
    GlossaryTerm,
    glossary_hash,
    load_glossary,
    protected_terms,
)


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "voxkit"
    / "data"
    / "glossary.example.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_glossary_parses_bundled_example() -> None:
    glossary = load_glossary(EXAMPLE_PATH)
    assert glossary.version == 1
    assert glossary.language == "zh"
    assert len(glossary.terms) == 4
    sources = [term.source for term in glossary.terms]
    assert sources == ["Claude", "Anthropic", "agent", "MCP"]


def test_load_glossary_rejects_duplicate_source(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "terms": [
            {"source": "Claude", "protected": True},
            {"source": "Claude", "target": "克劳德"},
        ],
    }
    path = tmp_path / "dup.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="duplicate glossary source: Claude"):
        load_glossary(path)


def test_glossary_hash_stable_across_term_reordering(tmp_path: Path) -> None:
    base_terms = [
        {"source": "Claude", "protected": True},
        {"source": "agent", "target": "智能体", "casePolicy": "smart"},
        {"source": "MCP", "target": "MCP", "protected": True, "casePolicy": "strict"},
    ]
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    _write_json(path_a, {"version": 1, "terms": base_terms})
    _write_json(
        path_b,
        {"version": 1, "terms": list(reversed(base_terms))},
    )

    hash_a = glossary_hash(load_glossary(path_a))
    hash_b = glossary_hash(load_glossary(path_b))
    assert hash_a == hash_b


def test_glossary_hash_changes_when_target_changes() -> None:
    g1 = Glossary(
        terms=[GlossaryTerm(source="agent", target="智能体")],
    )
    g2 = Glossary(
        terms=[GlossaryTerm(source="agent", target="代理")],
    )
    assert glossary_hash(g1) != glossary_hash(g2)


def test_glossary_hash_independent_of_version_and_language() -> None:
    terms = [GlossaryTerm(source="Claude", protected=True)]
    g1 = Glossary(version=1, language="zh", terms=terms)
    g2 = Glossary(version=2, language="en", terms=terms)
    assert glossary_hash(g1) == glossary_hash(g2)


def test_glossary_hash_is_64_char_lowercase_hex() -> None:
    glossary = load_glossary(EXAMPLE_PATH)
    h = glossary_hash(glossary)
    assert isinstance(h, str)
    assert len(h) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", h) is not None


def test_protected_terms_returns_expected_set() -> None:
    glossary = load_glossary(EXAMPLE_PATH)
    assert protected_terms(glossary) == {"Claude", "Anthropic", "MCP"}

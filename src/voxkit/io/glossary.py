"""Glossary：proofread / translate 阶段的可选输入。

用途：
  - 保护术语：``protected: true`` 表示模型不得改写该词（例如品牌名 ``Claude``）。
  - 指定译法：``target`` 给出 source → target 的固定映射；``None`` 表示只保护不替换。

``glossary_hash`` 是按"内容"算出的稳定 hex 摘要（sha256），写到 manifest 里
用作审计 / 复现键。**故意不**直接对文件字节哈希，这样：

  - 调整缩进 / 重排 ``terms`` 顺序 / 改 ``language`` 这种与语义无关的编辑
    不会触发下游 rerun；
  - 真正改了某个 term 的 ``source`` / ``target`` / ``protected`` / ``casePolicy``
    才会让 hash 变。

具体地，hash 只覆盖 ``terms`` 内容（按 ``source`` 升序、字段 ``sort_keys`` 序列化），
不包含 ``version`` / ``language`` / 文件级注释。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class GlossaryTerm(BaseModel):
    """单条术语条目。

    - ``source``：待保护或翻译的源词；同一份 glossary 内必须唯一。
    - ``target``：指定译法；``None`` 表示只保护不替换。
    - ``protected``：True 表示模型不得改写该词（proofread 风险判定会用到）。
    - ``case_policy``：
        - ``"strict"``：完全大小写匹配
        - ``"smart"``：句首 / 全大写情况自动适配（默认）
    """

    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(..., description="待保护或翻译的源词")
    target: Optional[str] = Field(None, description="指定译法；None 表示只保护不替换")
    protected: bool = Field(False, description="True 时模型不得改写")
    case_policy: Literal["strict", "smart"] = Field(
        "smart",
        alias="casePolicy",
        description='"strict" 完全大小写匹配；"smart" 句首/全大写自动适配',
    )


class Glossary(BaseModel):
    """Glossary 顶层模型，对应 ``glossary.json``。"""

    model_config = ConfigDict(populate_by_name=True)

    version: int = Field(1, description="glossary schema 版本")
    language: Optional[str] = Field(None, description="主语言代码（可选，仅作审计）")
    terms: List[GlossaryTerm]


def load_glossary(path: Union[Path, str]) -> Glossary:
    """从 JSON 文件加载 glossary。

    加载后会检查 ``terms[*].source`` 是否重复，重复时抛 ``ValueError``。
    """

    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    glossary = Glossary.model_validate(data)

    seen: set[str] = set()
    for term in glossary.terms:
        if term.source in seen:
            raise ValueError(f"duplicate glossary source: {term.source}")
        seen.add(term.source)

    return glossary


def glossary_hash(glossary: Glossary) -> str:
    """计算 glossary 的稳定内容哈希（64 字符小写 hex sha256）。

    实现要点：
      - 只 hash ``terms`` 内容，不包含 ``version`` / ``language``；
      - 每条 term 用 ``model_dump(by_alias=False, exclude_defaults=False)`` 统一形式；
      - 按 ``source`` 升序排，整体用 ``sort_keys=True`` 序列化以消除字段顺序影响；
      - ``ensure_ascii=False`` 让 CJK 字符直接进哈希，避免编码层多余抽象。
    """

    dumped = [
        term.model_dump(by_alias=False, exclude_defaults=False)
        for term in glossary.terms
    ]
    sorted_terms = sorted(dumped, key=lambda t: t["source"])
    payload = json.dumps(
        sorted_terms,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def protected_terms(glossary: Glossary) -> set[str]:
    """收集所有 ``protected=True`` 的 ``source`` 词集合。"""

    return {term.source for term in glossary.terms if term.protected}


__all__ = [
    "GlossaryTerm",
    "Glossary",
    "load_glossary",
    "glossary_hash",
    "protected_terms",
]

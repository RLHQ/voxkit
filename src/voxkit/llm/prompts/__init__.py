"""Prompt 模板加载器。

约定文件名 ``<stage>.<version>.md``，与本模块同目录。返回 ``(text, sha256)``
让上层把 hash 写进 manifest，满足"prompt version 可追踪"的审计要求。

在内存中做一次 caching：同一进程内重复加载不会重算 hash，也不会重读文件。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

from voxkit.llm.errors import LLMError

_PROMPT_DIR: Path = Path(__file__).resolve().parent

# (stage, version) → (text, sha256_hex)
_CACHE: Dict[Tuple[str, str], Tuple[str, str]] = {}


def load_prompt(stage: str, version: str = "v1") -> Tuple[str, str]:
    """加载 prompt 模板。

    Args:
        stage: 例如 ``"proofread"`` / ``"translate"``。
        version: 例如 ``"v1"``；与文件名第二段对齐。

    Returns:
        ``(template_text, sha256_hex)``：sha256 是文本（UTF-8 字节）的十六
        进制摘要，64 字符。

    Raises:
        LLMError: 文件缺失或读取失败。
    """
    key = (stage, version)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    path = _PROMPT_DIR / f"{stage}.{version}.md"
    if not path.is_file():
        raise LLMError(
            f"prompt not found: {path.name} (expected at {path})"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise LLMError(f"failed to read prompt {path}: {e}") from e

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _CACHE[key] = (text, digest)
    return text, digest


__all__ = ["load_prompt"]

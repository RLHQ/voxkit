"""Provider 注册表的契约测试。"""

from __future__ import annotations

import pytest

from voxkit.llm.errors import LLMError
from voxkit.llm.providers import PROVIDERS, get_provider


def test_deepseek_spec_shape() -> None:
    spec = get_provider("deepseek")
    assert spec.name == "deepseek"
    assert spec.base_url == "https://api.deepseek.com"
    assert spec.api_key_env == "DEEPSEEK_API_KEY"
    assert spec.default_model == "deepseek-v4-flash"
    assert spec.max_context_tokens == 64000


def test_deepseek_max_context_tokens_constant() -> None:
    # 直接读 dict 也是合法访问路径（registry 是公开的）。
    assert PROVIDERS["deepseek"].max_context_tokens == 64000


def test_unknown_provider_lists_available() -> None:
    with pytest.raises(LLMError) as exc:
        get_provider("nope")
    msg = str(exc.value)
    assert "nope" in msg
    # 错误信息应当列出已注册 provider，便于定位拼写错误。
    assert "deepseek" in msg

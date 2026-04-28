"""voxkit.core.env — patched_env 行为测试。

重点：
- 模型 cache 齐全时自动注入 ``HF_HUB_OFFLINE=1``（Fix 1）
- 用户已显式设 ``HF_HUB_OFFLINE`` → 不覆盖
- ``extra`` dict 同名 key 优先于自动注入
- macOS DYLD_LIBRARY_PATH prepend（保留原有行为）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voxkit.core import env as core_env


# ── helpers ───────────────────────────────────────────────────────────
def _stub_models_ready(monkeypatch, ready: bool) -> None:
    """直接 patch core.env 命名空间里 lazy import 后的 models_offline_ready；
    更稳：patch 源头函数，让 lazy import 也拿到 stub。"""
    monkeypatch.setattr(
        "voxkit.core.bundle.models_offline_ready",
        lambda hub=None: ready,
    )


def _clear_hf_env(monkeypatch) -> None:
    for k in ("HF_HUB_OFFLINE", "HF_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(k, raising=False)


# ── HF_HUB_OFFLINE 自动注入 ───────────────────────────────────────────
def test_patched_env_injects_offline_when_cache_ready(monkeypatch):
    """cache 齐全 + 用户没显式设 HF_HUB_OFFLINE → 自动 set 1。"""
    _clear_hf_env(monkeypatch)
    _stub_models_ready(monkeypatch, True)

    env = core_env.patched_env()
    assert env.get("HF_HUB_OFFLINE") == "1"


def test_patched_env_no_offline_when_cache_missing(monkeypatch):
    """cache 不齐全 → 不注入（让 worker 走在线下载逻辑）。"""
    _clear_hf_env(monkeypatch)
    _stub_models_ready(monkeypatch, False)

    env = core_env.patched_env()
    assert "HF_HUB_OFFLINE" not in env


def test_patched_env_respects_user_explicit_offline_zero(monkeypatch):
    """用户显式设 HF_HUB_OFFLINE=0（强制在线调试）→ 不覆盖。

    哪怕 cache 齐全，也尊重用户的逃生口（dev 想测在线行为）。
    """
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    _stub_models_ready(monkeypatch, True)

    env = core_env.patched_env()
    assert env.get("HF_HUB_OFFLINE") == "0", "用户的显式 0 必须保留"


def test_patched_env_respects_user_explicit_offline_one(monkeypatch):
    """用户显式设 HF_HUB_OFFLINE=1 + cache 不齐全 → 仍然保留 1（用户知道自己在干嘛）。"""
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _stub_models_ready(monkeypatch, False)

    env = core_env.patched_env()
    assert env.get("HF_HUB_OFFLINE") == "1"


def test_patched_env_extra_overrides_auto_offline(monkeypatch):
    """``extra`` dict 同名 key 优先于自动注入。"""
    _clear_hf_env(monkeypatch)
    _stub_models_ready(monkeypatch, True)

    env = core_env.patched_env(extra={"HF_HUB_OFFLINE": "0"})
    assert env.get("HF_HUB_OFFLINE") == "0"


def test_patched_env_extra_passthrough(monkeypatch):
    """``extra`` 中的其他 key 正常注入，不影响自动 OFFLINE 决策。"""
    _clear_hf_env(monkeypatch)
    _stub_models_ready(monkeypatch, True)

    env = core_env.patched_env(extra={"FOO": "bar"})
    assert env.get("FOO") == "bar"
    assert env.get("HF_HUB_OFFLINE") == "1"


# ── DYLD_LIBRARY_PATH（保留原有行为）─────────────────────────────────
def test_patched_env_prepends_dyld_when_lib_dir_found(monkeypatch):
    """macOS 上找到 ffmpeg lib → prepend 到 DYLD_LIBRARY_PATH。"""
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(core_env, "find_ffmpeg_lib_dir", lambda: "/fake/lib")
    _stub_models_ready(monkeypatch, False)  # 隔离 OFFLINE 逻辑

    env = core_env.patched_env()
    assert env.get("DYLD_LIBRARY_PATH") == "/fake/lib"


def test_patched_env_prepends_dyld_preserves_existing(monkeypatch):
    """已有 DYLD_LIBRARY_PATH → 新 lib_dir 拼在前面，原值保留。"""
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/existing/path")
    monkeypatch.setattr(core_env, "find_ffmpeg_lib_dir", lambda: "/fake/lib")
    _stub_models_ready(monkeypatch, False)

    env = core_env.patched_env()
    assert env.get("DYLD_LIBRARY_PATH") == "/fake/lib:/existing/path"


def test_patched_env_no_dyld_change_when_no_lib_dir(monkeypatch):
    """非 macOS 或找不到 lib → DYLD_LIBRARY_PATH 不改。"""
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(core_env, "find_ffmpeg_lib_dir", lambda: None)
    _stub_models_ready(monkeypatch, False)

    env = core_env.patched_env()
    assert "DYLD_LIBRARY_PATH" not in env

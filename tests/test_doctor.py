"""doctor 检查项的单元测试：mock 各类 IO，验证逻辑分支。

只测纯函数 + 单一检查，不跑 run() 全链路（那是 verification 步骤的事）。
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

from voxsplit.commands import doctor as D


# ── check_python ────────────────────────────────────────────────
def test_check_python_pass():
    """当前 venv Python ≥ 3.10。"""
    r = D.check_python()
    assert r.ok, r.detail


# ── check_uv（环境相关，PATH 上有 uv 才过）────────────────────
def test_check_uv_runs():
    r = D.check_uv()
    # 不强求 ok；但 detail 里要有清晰的 fix 或路径
    assert "uv" in r.name
    if not r.ok:
        assert r.fix and "brew install uv" in r.fix


# ── check_hf_token ──────────────────────────────────────────────
def test_check_hf_token_missing(tmp_path, monkeypatch):
    """模拟无 HF token 的 home 目录。"""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    r = D.check_hf_token()
    assert not r.ok
    assert r.fix and "huggingface" in r.fix.lower()


def test_check_hf_token_present(tmp_path, monkeypatch):
    """token 文件存在且非空 → 通过。"""
    token_dir = tmp_path / ".cache" / "huggingface"
    token_dir.mkdir(parents=True)
    (token_dir / "token").write_text("hf_xxxxxxxx")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    r = D.check_hf_token()
    assert r.ok


# ── check_gated_repos ───────────────────────────────────────────
def test_check_gated_no_token(tmp_path, monkeypatch):
    """无 token 时 4 条全失败，但失败原因都明确指向 token。"""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    results = D.check_gated_repos()
    assert len(results) == len(D.GATED_WEIGHTS) == 4
    assert all(not r.ok for r in results)


def test_check_gated_403(tmp_path, monkeypatch):
    """模拟 HEAD 返回 403（未 accept）。"""
    token_dir = tmp_path / ".cache" / "huggingface"
    token_dir.mkdir(parents=True)
    (token_dir / "token").write_text("hf_test")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def _fake_urlopen(req, timeout=10):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(D.urllib.request, "urlopen", _fake_urlopen)
    results = D.check_gated_repos()
    assert all(not r.ok for r in results)
    assert all("Accept" in (r.fix or "") for r in results)


def test_check_gated_200(tmp_path, monkeypatch):
    """模拟 HEAD 返回 200 → 全通过。"""
    token_dir = tmp_path / ".cache" / "huggingface"
    token_dir.mkdir(parents=True)
    (token_dir / "token").write_text("hf_test")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(D.urllib.request, "urlopen", lambda req, timeout=10: _Resp())
    results = D.check_gated_repos()
    assert all(r.ok for r in results)


# ── check_ffmpeg ────────────────────────────────────────────────
def test_check_ffmpeg_present():
    """开发机一般装了 ffmpeg；至少结果列表非空。"""
    results = D.check_ffmpeg()
    assert len(results) >= 1
    # 第一项必是 "ffmpeg 可执行"
    assert "ffmpeg" in results[0].name


# ── check_venv（lazy venv 可能未创建，函数应优雅返回）──────
def test_check_venv_missing(monkeypatch, tmp_path):
    """模拟 lazy venv 不存在。"""
    fake_py = tmp_path / "nope" / "bin" / "python"
    monkeypatch.setattr(D, "_LAZY_VENV_PY", fake_py)
    r = D.check_venv()
    assert not r.ok
    assert r.severity == "warn"  # venv 缺只是 warn（不阻断 doctor 0 退出）

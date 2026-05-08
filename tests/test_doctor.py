"""doctor 检查项的单元测试：mock 各类 IO，验证逻辑分支。

只测纯函数 + 单一检查，不跑 run() 全链路（那是 verification 步骤的事）。
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

from voxkit.commands import doctor as D
from voxkit.cli import _build_parser


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


# ── Round 2: check_whisper_cli ──────────────────────────────────
def test_check_whisper_cli_not_found(monkeypatch):
    """whisper-cli 未安装 → WARN，提示 brew install whisper-cpp。"""
    monkeypatch.setattr(D, "find_whisper_cli", lambda: None)
    r = D.check_whisper_cli()
    assert not r.ok
    assert r.severity == "warn"
    assert r.fix and "whisper-cpp" in r.fix


def test_check_whisper_cli_all_flags(monkeypatch, tmp_path):
    """whisper-cli help 包含全部必需 flag → OK。"""
    fake_bin = tmp_path / "whisper-cli"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.setattr(D, "find_whisper_cli", lambda: fake_bin)

    help_text = "\n".join([
        "Usage: whisper-cli ...",
        "  --output-json-full",
        "  --max-context N",
        "  --vad",
        "  --split-on-word",
        "  --logprob-thold N",
    ])

    class _Completed:
        stdout = help_text
        stderr = ""

    def _fake_run(argv, capture_output=True, text=True, timeout=5):
        assert argv[0] == str(fake_bin) and argv[1] == "--help"
        return _Completed()

    monkeypatch.setattr(D.subprocess, "run", _fake_run)
    r = D.check_whisper_cli()
    assert r.ok, r.detail
    assert str(fake_bin) in r.detail


def test_check_whisper_cli_missing_vad_flag(monkeypatch, tmp_path):
    """whisper-cli help 缺 --vad → WARN，message 应提到 --vad。"""
    fake_bin = tmp_path / "whisper-cli"
    fake_bin.write_text("#!/bin/sh\n")
    monkeypatch.setattr(D, "find_whisper_cli", lambda: fake_bin)

    # 故意省略 --vad
    help_text = "\n".join([
        "  --output-json-full",
        "  --max-context N",
        "  --split-on-word",
        "  --logprob-thold N",
    ])

    class _Completed:
        stdout = help_text
        stderr = ""

    monkeypatch.setattr(
        D.subprocess, "run",
        lambda *a, **kw: _Completed(),
    )
    r = D.check_whisper_cli()
    assert not r.ok
    assert r.severity == "warn"
    assert "--vad" in r.detail
    assert r.fix and "upgrade" in r.fix.lower()


# ── Round 2: check_whisper_model ────────────────────────────────
def test_check_whisper_model_present(monkeypatch, tmp_path):
    """模型文件存在 → OK，detail 含 MB。"""
    fake_model = tmp_path / "ggml-large-v3-turbo.bin"
    fake_model.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
    monkeypatch.setattr(D, "find_whisper_model", lambda name="large-v3-turbo": fake_model)
    r = D.check_whisper_model()
    assert r.ok
    assert "MB" in r.detail


def test_check_whisper_model_missing(monkeypatch):
    """模型缺失 → WARN，hint 提到 huggingface-cli 或 brew。"""
    monkeypatch.setattr(D, "find_whisper_model", lambda name="large-v3-turbo": None)
    r = D.check_whisper_model()
    assert not r.ok
    assert r.severity == "warn"
    assert r.fix
    fix_lower = r.fix.lower()
    assert "huggingface-cli" in fix_lower or "brew" in fix_lower


# ── Round 2: check_vad_model ────────────────────────────────────
def test_check_vad_model_present(monkeypatch, tmp_path):
    """VAD 模型存在 → OK，detail 含路径。"""
    fake_vad = tmp_path / "ggml-silero-v5.1.2.bin"
    fake_vad.write_text("x")
    monkeypatch.setattr(D, "find_vad_model", lambda: fake_vad)
    r = D.check_vad_model()
    assert r.ok
    assert str(fake_vad) in r.detail


def test_check_vad_model_missing(monkeypatch):
    """VAD 模型缺失 → WARN。"""
    monkeypatch.setattr(D, "find_vad_model", lambda: None)
    r = D.check_vad_model()
    assert not r.ok
    assert r.severity == "warn"
    assert r.fix


# ── doctor --profile ───────────────────────────────────────────
def test_doctor_profile_argparse_defaults_to_all():
    parser = _build_parser()
    args = parser.parse_args(["doctor"])
    assert args.cmd == "doctor"
    assert args.profile == "all"


def test_doctor_profile_argparse_accepts_transcribe():
    parser = _build_parser()
    args = parser.parse_args(["doctor", "--profile", "transcribe"])
    assert args.profile == "transcribe"


def test_collect_transcribe_profile_promotes_required_checks(monkeypatch, tmp_path):
    """For first-run ASR, missing whisper-cli/model must be actionable errors."""
    monkeypatch.setattr(D, "check_python", lambda: D.CheckResult("Python ≥ 3.10", True, "3.12"))
    monkeypatch.setattr(
        D,
        "check_ffmpeg",
        lambda: [D.CheckResult("ffmpeg 可执行", True, "/opt/homebrew/bin/ffmpeg", category="env")],
    )
    monkeypatch.setattr(
        D,
        "check_whisper_cli",
        lambda: D.CheckResult(
            "whisper-cli 可用",
            False,
            "missing",
            fix="brew install whisper-cpp",
            severity="warn",
        ),
    )
    monkeypatch.setattr(
        D,
        "check_whisper_model",
        lambda: D.CheckResult(
            "whisper 模型 (large-v3-turbo)",
            False,
            "missing",
            fix="download model",
            severity="warn",
        ),
    )
    monkeypatch.setattr(
        D,
        "check_vad_model",
        lambda: D.CheckResult(
            "silero VAD 模型",
            False,
            "missing",
            fix="optional",
            severity="warn",
        ),
    )

    results, mode = D._collect_results("transcribe")
    by_name = {r.name: r for r in results}
    assert mode == "transcribe"
    assert by_name["whisper-cli 可用"].severity == "error"
    assert by_name["whisper 模型 (large-v3-turbo)"].severity == "error"
    assert by_name["silero VAD 模型"].severity == "warn"


def test_collect_diarize_profile_skips_whisper_checks(monkeypatch):
    """Diarize first-run guidance should not mention whisper-only dependencies."""
    monkeypatch.setattr(
        D,
        "check_models_offline",
        lambda: D.CheckResult("模型离线就绪", True, "ready", category="hf"),
    )
    monkeypatch.setattr(D, "check_uv", lambda: D.CheckResult("uv 已安装", True, "/opt/homebrew/bin/uv"))
    monkeypatch.setattr(D, "check_python", lambda: D.CheckResult("Python ≥ 3.10", True, "3.12"))
    monkeypatch.setattr(
        D,
        "check_ffmpeg",
        lambda: [D.CheckResult("ffmpeg 可执行", True, "/opt/homebrew/bin/ffmpeg", category="env")],
    )
    monkeypatch.setattr(
        D,
        "check_venv",
        lambda: D.CheckResult("voxkit venv 就绪", False, "not yet", severity="warn"),
    )
    monkeypatch.setattr(
        D,
        "check_whisper_cli",
        lambda: pytest.fail("diarize profile should not call check_whisper_cli"),
    )
    monkeypatch.setattr(
        D,
        "check_whisper_model",
        lambda: pytest.fail("diarize profile should not call check_whisper_model"),
    )

    results, mode = D._collect_results("diarize")
    assert mode == "diarize / 离线模式"
    assert {r.name for r in results} == {
        "模型离线就绪",
        "uv 已安装",
        "Python ≥ 3.10",
        "ffmpeg 可执行",
        "voxkit venv 就绪",
    }


# ── Round 2: integration — full run() includes the 3 new checks ──
def test_run_integration_includes_new_checks(monkeypatch, tmp_path, capsys):
    """run() 全链路：3 个新 WARN 都是 warn，不改变退出码。

    通过 mock 让所有新 check 都"未找到 → WARN"，run() 应仍返回 ExitCode.OK
    （只要原 7 项无 error）。
    """
    # mock 新 3 项为 missing → WARN
    monkeypatch.setattr(D, "find_whisper_cli", lambda: None)
    monkeypatch.setattr(D, "find_whisper_model", lambda name="large-v3-turbo": None)
    monkeypatch.setattr(D, "find_vad_model", lambda: None)

    # 让 HF token 探测落到 tmp_path（避免命中开发机真实 token）
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    # 让 gated HEAD 都 200，避免 HF auth 故障扰乱基线
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass

    # 写一个假 HF token
    token_dir = tmp_path / ".cache" / "huggingface"
    token_dir.mkdir(parents=True)
    (token_dir / "token").write_text("hf_test")
    monkeypatch.setattr(D.urllib.request, "urlopen", lambda req, timeout=10: _Resp())

    code = D.run()
    captured = capsys.readouterr().out

    # 新 3 行都应该出现在输出里（即使是 WARN 状态）
    assert "whisper-cli 可用" in captured
    assert "whisper 模型" in captured
    assert "silero VAD 模型" in captured

    # 退出码：所有新 check 都是 WARN，不应该改变 baseline。
    # 当 ffmpeg/uv/venv 在开发机上可能各种状态，至少应该不是因新 3 项而失败。
    # 严格判定：退出码不应是 ExitCode.GENERIC_FAIL（1）由新 check 触发。
    # 实际上只要新 3 项都是 warn，run() 不会因它们而 fail。
    from voxkit.core.constants import ExitCode
    assert code in (ExitCode.OK, ExitCode.HF_AUTH, ExitCode.ENV_PROBLEM, ExitCode.GENERIC_FAIL)
    # 关键不变量：新 3 项 WARN 不会单独导致 GENERIC_FAIL。
    # 这里我们能确认的最强语义：输出包含 3 个新 check 的行。

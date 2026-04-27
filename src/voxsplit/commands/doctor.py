"""voxsplit doctor — 6 项自检。

每项独立函数返回 (ok: bool, label: str, fix_hint: str)。
全绿 exit 0；任一失败 exit 2（HF）/ 3（环境）/ 1（其他）。
"""

from __future__ import annotations

import concurrent.futures
import platform
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from voxsplit.core import env as core_env
from voxsplit.core import audio as core_audio
from voxsplit.core.constants import (
    GATED_WEIGHTS,
    HF_TOKEN_PATHS,
    VENV_PYTHON,
    ExitCode,
)


_LAZY_VENV_PY = VENV_PYTHON  # 测试可 monkey-patch 替换


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: Optional[str] = None
    severity: str = "error"  # error | warn
    category: str = "generic"  # generic | hf | env — 决定 exit code 路由


# ── 1. uv ───────────────────────────────────────────────────────
def check_uv() -> CheckResult:
    p = shutil.which("uv")
    if p:
        try:
            ver = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=3).stdout.strip()
        except Exception:
            ver = "(version 探测失败)"
        return CheckResult("uv 已安装", True, f"{p} ({ver})")
    return CheckResult(
        "uv 已安装", False,
        "PATH 中未找到 uv",
        fix="brew install uv",
    )


# ── 2. Python ≥ 3.10 ────────────────────────────────────────────
def check_python() -> CheckResult:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    detail = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if ok:
        return CheckResult("Python ≥ 3.10", True, detail)
    return CheckResult(
        "Python ≥ 3.10", False, detail,
        fix="使用 uv 安装 3.12: uv python install 3.12",
    )


# ── 3. HF token ────────────────────────────────────────────────
def _read_hf_token() -> tuple[Optional[str], Optional[Path]]:
    """返回 (token, 找到 token 的文件路径)；都没有则 (None, None)。"""
    # 注意：每次重新读 Path.home()，让 monkey-patch 生效（测试用）
    for p in [Path.home() / rel for rel in [
        Path(".cache/huggingface/token"),
        Path(".huggingface/token"),
    ]]:
        if p.is_file():
            t = p.read_text().strip()
            if t:
                return t, p
    return None, None


def check_hf_token() -> CheckResult:
    token, path = _read_hf_token()
    if token:
        return CheckResult("HF token 存在", True, str(path), category="hf")
    return CheckResult(
        "HF token 存在", False,
        "未找到 ~/.cache/huggingface/token 或 ~/.huggingface/token",
        fix=("1) https://huggingface.co/settings/tokens 创建 token\n"
             "  2) huggingface-cli login（venv 内）或写入 ~/.cache/huggingface/token"),
        category="hf",
    )


# ── 4. 4 个 gated repo accept ──────────────────────────────────
def _head_one_gated(token: str, name: str, url: str) -> CheckResult:
    """单个 HEAD：被 ThreadPoolExecutor 并发调度。

    - 200 → accepted
    - 401/403 → 未 accept 或 token 不对
    - 其他码或网络错误 → warn，不阻断
    """
    headers = {"Authorization": f"Bearer {token}"}
    label = f"gated accept: {name}"
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:
        return CheckResult(
            label, False, f"HEAD 失败: {e}",
            fix=f"网络问题或 token 失效；浏览器打开 https://huggingface.co/{name}",
            category="hf",
        )
    if code == 200:
        return CheckResult(label, True, "HEAD 200", category="hf")
    if code in (401, 403):
        return CheckResult(
            label, False,
            f"HEAD {code}（未 accept 或 token 不匹配）",
            fix=f"https://huggingface.co/{name} 点 Accept",
            category="hf",
        )
    return CheckResult(
        label, False,
        f"HEAD 异常码: {code}",
        fix=f"重试或浏览器访问 https://huggingface.co/{name}",
        severity="warn", category="hf",
    )


def check_gated_repos() -> List[CheckResult]:
    """4 个 gated repo HEAD 并发执行（最坏总耗时 ≈ 单次 timeout 10s）。"""
    token, _ = _read_hf_token()
    if not token:
        return [
            CheckResult(
                f"gated accept: {name}", False,
                "HF token 缺失，无法 HEAD",
                fix="先解决 HF token 检查",
                category="hf",
            )
            for name, _ in GATED_WEIGHTS
        ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GATED_WEIGHTS)) as ex:
        return list(ex.map(lambda nu: _head_one_gated(token, nu[0], nu[1]), GATED_WEIGHTS))


# ── 5. ffmpeg / libavutil ──────────────────────────────────────
def check_ffmpeg() -> List[CheckResult]:
    results: List[CheckResult] = []

    ffmpeg = core_audio.find_ffmpeg()
    if not ffmpeg:
        results.append(CheckResult(
            "ffmpeg 可执行", False,
            "PATH / 常见路径均未找到",
            fix="brew install ffmpeg-full（macOS）或 apt install ffmpeg",
            category="env",
        ))
        return results
    results.append(CheckResult("ffmpeg 可执行", True, ffmpeg, category="env"))

    major = core_audio.get_ffmpeg_major_version()
    if major is None:
        results.append(CheckResult(
            "ffmpeg 版本兼容", False,
            "ffmpeg -version 解析失败",
            severity="warn", category="env",
        ))
    elif 4 <= major <= 8:
        results.append(CheckResult("ffmpeg 版本兼容", True, f"major={major}", category="env"))
    else:
        results.append(CheckResult(
            "ffmpeg 版本兼容", False,
            f"major={major} 不在支持区间 [4, 8]",
            fix="切换到 ffmpeg 4-8（torchcodec 兼容范围）",
            severity="warn", category="env",
        ))

    if platform.system() == "Darwin":
        lib_dir = core_env.find_ffmpeg_lib_dir()
        if lib_dir:
            results.append(CheckResult(
                "libavutil 可定位（DYLD）", True,
                f"{lib_dir}（voxsplit 会自动 export DYLD_LIBRARY_PATH）",
                category="env",
            ))
        else:
            results.append(CheckResult(
                "libavutil 可定位（DYLD）", False,
                "macOS 上未在常见路径找到 libavutil*.dylib",
                fix="brew install ffmpeg-full",
                category="env",
            ))
    return results


# ── 6. lazy venv 与 pyannote.audio 版本 ────────────────────────
def check_venv() -> CheckResult:
    if not _LAZY_VENV_PY.is_file():
        return CheckResult(
            "voxsplit venv 就绪", False,
            f"{_LAZY_VENV_PY} 不存在",
            fix="运行 voxsplit setup（或首次 voxsplit diarize 自动触发）",
            severity="warn",
        )
    try:
        out = subprocess.run(
            [str(_LAZY_VENV_PY), "-c", "import pyannote.audio; print(pyannote.audio.__version__)"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        return CheckResult(
            "voxsplit venv 就绪", False,
            f"venv python 调用失败: {e}",
            fix="rm -rf ~/.local/share/voxsplit 后重跑 voxsplit setup",
        )
    if out.returncode != 0:
        return CheckResult(
            "voxsplit venv 就绪", False,
            f"pyannote.audio 未安装（{out.stderr.strip()[:200]}）",
            fix="voxsplit setup",
        )
    ver = out.stdout.strip()
    return CheckResult("voxsplit venv 就绪", True, f"pyannote.audio {ver}")


# ── 主流程 ──────────────────────────────────────────────────────
def _print_result(r: CheckResult) -> None:
    icon = "✅" if r.ok else ("⚠️ " if r.severity == "warn" else "❌")
    print(f"{icon} {r.name}: {r.detail}")
    if not r.ok and r.fix:
        for line in r.fix.splitlines():
            print(f"   ↳ {line}")


def run() -> int:
    """跑全部 6 类检查并打印；返回 exit code。"""
    all_checks: List[Callable[[], object]] = [
        check_uv,
        check_python,
        check_hf_token,
        check_gated_repos,
        check_ffmpeg,
        check_venv,
    ]

    results: List[CheckResult] = []
    for fn in all_checks:
        out = fn()
        if isinstance(out, list):
            results.extend(out)
        else:
            results.append(out)

    print("voxsplit doctor")
    print("=" * 50)
    for r in results:
        _print_result(r)
    print("=" * 50)

    failed = [r for r in results if not r.ok and r.severity != "warn"]
    warned = [r for r in results if not r.ok and r.severity == "warn"]
    if failed:
        categories = {r.category for r in failed}
        if "hf" in categories:
            print(f"\n❌ {len(failed)} 项失败（含 HF 认证）")
            return ExitCode.HF_AUTH
        if "env" in categories:
            print(f"\n❌ {len(failed)} 项失败（含环境问题）")
            return ExitCode.ENV_PROBLEM
        print(f"\n❌ {len(failed)} 项失败")
        return ExitCode.GENERIC_FAIL
    if warned:
        print(f"\n⚠️  全部关键项通过；{len(warned)} 项警告（不阻断 diarize）")
        return ExitCode.OK
    print("\n✅ 全绿。")
    return ExitCode.OK


__all__ = ["run", "check_uv", "check_python", "check_hf_token",
           "check_gated_repos", "check_ffmpeg", "check_venv",
           "CheckResult"]

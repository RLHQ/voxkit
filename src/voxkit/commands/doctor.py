"""voxkit doctor — 6 项自检。

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

from voxkit.core import env as core_env
from voxkit.core import audio as core_audio
from voxkit.core import bundle as core_bundle
from voxkit.core.constants import (
    BUNDLE_MODELS,
    GATED_WEIGHTS,
    HF_TOKEN_PATHS,
    VENV_PYTHON,
    ExitCode,
)
from voxkit.core.whisper_exec import (
    find_whisper_cli,
    find_whisper_model,
    find_vad_model,
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
                f"{lib_dir}（voxkit 会自动 export DYLD_LIBRARY_PATH）",
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


# ── 6. 模型离线就绪（命中则跳过 HF token + gated 网络检查）─────
def check_models_offline() -> CheckResult:
    """检测 HF cache 中 4 个模型是否都齐全。

    完整 → ✅，doctor 切换"离线模式"显示，diarize worker 自动获得
    ``HF_HUB_OFFLINE=1`` 注入（见 ``core.env.patched_env``）。
    任一缺失 → 不阻断，由后续 gated 检查接力。

    判定标准与 ``core.bundle.models_offline_ready`` 完全一致——后者用于纯布尔决策
    （patched_env 注入逻辑），这里需要额外列出**具体缺失的仓库名**用于人类提示，
    所以保留独立 enumerate 而非套壳调用。
    """
    hub = core_bundle.hf_hub_cache_dir()
    missing: List[str] = []
    for spec in BUNDLE_MODELS:
        repo_id = spec["repo_id"]
        d = hub / core_bundle.repo_id_to_dirname(repo_id)
        refs = d / "refs" / "main"
        if not refs.is_file():
            missing.append(repo_id)
            continue
        commit = refs.read_text().strip()
        if not commit:
            missing.append(repo_id)
            continue
        snap = d / "snapshots" / commit
        if not snap.is_dir() or not any(snap.iterdir()):
            missing.append(repo_id)

    if missing:
        return CheckResult(
            "模型离线就绪", False,
            f"缺 {len(missing)} 个：{', '.join(repo_id_short(r) for r in missing)}",
            fix="voxkit fetch-bundle  （或先跑 diarize 触发 HF 下载）",
            severity="warn", category="hf",
        )
    return CheckResult(
        "模型离线就绪", True,
        f"4 个模型齐全 ({hub}); diarize 将自动 HF_HUB_OFFLINE=1",
        category="hf",
    )


def repo_id_short(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1]


# ── 7. lazy venv 与 pyannote.audio 版本 ────────────────────────
def check_venv() -> CheckResult:
    if not _LAZY_VENV_PY.is_file():
        return CheckResult(
            "voxkit venv 就绪", False,
            f"{_LAZY_VENV_PY} 不存在",
            fix="运行 voxkit setup（或首次 voxkit diarize 自动触发）",
            severity="warn",
        )
    try:
        out = subprocess.run(
            [str(_LAZY_VENV_PY), "-c", "import pyannote.audio; print(pyannote.audio.__version__)"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        return CheckResult(
            "voxkit venv 就绪", False,
            f"venv python 调用失败: {e}",
            fix="rm -rf ~/.local/share/voxkit 后重跑 voxkit setup",
        )
    if out.returncode != 0:
        return CheckResult(
            "voxkit venv 就绪", False,
            f"pyannote.audio 未安装（{out.stderr.strip()[:200]}）",
            fix="voxkit setup",
        )
    ver = out.stdout.strip()
    return CheckResult("voxkit venv 就绪", True, f"pyannote.audio {ver}")


# ── 8. whisper-cli 可用性 + 必需 flag ──────────────────────────
# 来自 Plan §K.1：transcribe pipeline 需要 whisper-cli 支持的 flag 集合。
_WHISPER_REQUIRED_FLAGS = (
    "--output-json-full",
    "--max-context",
    "--vad",
    "--split-on-word",
    "--logprob-thold",
)


def check_whisper_cli() -> CheckResult:
    """发现 whisper-cli；命中后跑 ``--help`` 校验关键 flag 是否齐全。

    Outcomes:
      - 未发现 whisper-cli → WARN（transcribe 才需要；diarize 不依赖）
      - 发现但 ``--help`` 缺关键 flag → WARN（建议升级）
      - 全部命中 → OK，detail 为可执行路径

    永远 WARN，不会以 error 阻断 diarize-only 用户。
    """
    bin_path = find_whisper_cli()
    if bin_path is None:
        return CheckResult(
            "whisper-cli 可用",
            False,
            "PATH / 常见路径未找到 whisper-cli",
            fix="voxkit transcribe 需要 whisper-cli；brew install whisper-cpp",
            severity="warn",
        )

    try:
        out = subprocess.run(
            [str(bin_path), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as e:
        return CheckResult(
            "whisper-cli 可用",
            False,
            f"{bin_path}（--help 调用失败: {e}）",
            fix="brew upgrade whisper-cpp",
            severity="warn",
        )

    help_text = (out.stdout or "") + "\n" + (out.stderr or "")
    missing = [flag for flag in _WHISPER_REQUIRED_FLAGS if flag not in help_text]
    if missing:
        return CheckResult(
            "whisper-cli 可用",
            False,
            f"{bin_path}（缺 flag: {', '.join(missing)}）",
            fix="whisper-cli 版本过旧；brew upgrade whisper-cpp",
            severity="warn",
        )

    return CheckResult(
        "whisper-cli 可用",
        True,
        f"{bin_path}（关键 flag 齐全）",
    )


# ── 9. whisper 主模型（large-v3-turbo）────────────────────────
def check_whisper_model() -> CheckResult:
    """查找默认 whisper.cpp ggml 模型 ``large-v3-turbo``。

    Outcomes:
      - 命中 → OK，附文件大小 (MB)
      - 未命中 → WARN，给出两条修复建议
    """
    p = find_whisper_model("large-v3-turbo")
    if p is not None:
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            detail = f"{p} ({size_mb:.0f} MB)"
        except OSError:
            detail = str(p)
        return CheckResult("whisper 模型 (large-v3-turbo)", True, detail)
    return CheckResult(
        "whisper 模型 (large-v3-turbo)",
        False,
        "未在 ~/.cache/voxkit/models / brew share 路径下找到",
        fix=(
            "huggingface-cli download ggerganov/whisper.cpp "
            "ggml-large-v3-turbo.bin --local-dir ~/.cache/voxkit/models\n"
            "  或 brew install whisper-cpp（自带 /opt/homebrew/share/whisper-cpp/）"
        ),
        severity="warn",
    )


# ── 10. silero VAD 模型 ───────────────────────────────────────
def check_vad_model() -> CheckResult:
    """查找 silero VAD 模型。

    Outcomes:
      - 命中 → OK，附路径
      - 未命中 → WARN（transcribe 仍可工作，只是不带 VAD）
    """
    p = find_vad_model()
    if p is not None:
        return CheckResult("silero VAD 模型", True, str(p))
    return CheckResult(
        "silero VAD 模型",
        False,
        "未找到 ggml-silero-v5.1.2.bin",
        fix=(
            "VAD 模型缺失 — voxkit transcribe 将以无 VAD 模式运行；"
            "可 brew install whisper-cpp 获取"
        ),
        severity="warn",
    )


# ── 主流程 ──────────────────────────────────────────────────────
def _print_result(r: CheckResult) -> None:
    icon = "✅" if r.ok else ("⚠️ " if r.severity == "warn" else "❌")
    print(f"{icon} {r.name}: {r.detail}")
    if not r.ok and r.fix:
        for line in r.fix.splitlines():
            print(f"   ↳ {line}")


def run() -> int:
    """跑全部检查并打印；返回 exit code。

    模型离线就绪时，HF token + 4 个 gated HEAD 都视为非必要（降级为 info），
    因为 pyannote 在 cache 命中时不发起任何网络请求。
    """
    # 先做离线检查决定后续模式
    offline = check_models_offline()
    offline_ready = offline.ok

    all_checks: List[Callable[[], object]] = [
        check_uv,
        check_python,
    ]
    # 离线就绪 → 跳过 HF token + gated repo（保留为 info 不阻断）
    if not offline_ready:
        all_checks += [check_hf_token, check_gated_repos]
    all_checks += [check_ffmpeg, check_venv]
    # Round 2：transcribe 相关（永 WARN，不阻断 diarize-only 用户）
    all_checks += [check_whisper_cli, check_whisper_model, check_vad_model]

    results: List[CheckResult] = [offline]
    for fn in all_checks:
        out = fn()
        if isinstance(out, list):
            results.extend(out)
        else:
            results.append(out)

    print("voxkit doctor")
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
        mode = "离线" if offline_ready else "在线"
        print(f"\n⚠️  全部关键项通过（{mode}模式）；{len(warned)} 项警告（不阻断 diarize）")
        return ExitCode.OK
    mode = "离线" if offline_ready else "在线"
    print(f"\n✅ 全绿（{mode}模式）。")
    return ExitCode.OK


__all__ = ["run", "check_uv", "check_python", "check_hf_token",
           "check_gated_repos", "check_ffmpeg", "check_venv",
           "check_models_offline",
           "check_whisper_cli", "check_whisper_model", "check_vad_model",
           "CheckResult"]

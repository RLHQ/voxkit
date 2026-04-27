"""voxsplit 用户级 lazy install。

设计：
- venv 路径：~/.local/share/voxsplit/venv （不污染项目目录）
- 标记文件：~/.cache/voxsplit/.installed （记录 pyannote 版本）
- 当前 voxsplit 源码以 editable 安装到 venv，使 worker 子进程能 `python -m voxsplit.core.pipeline`
- 模型缓存复用 HF 默认路径 ~/.cache/huggingface/hub/

退出策略：
- venv 已存在且 marker 版本匹配 → 跳过
- 不匹配或不存在 → 用 uv 创建 venv + uv pip install
- 失败抛 SetupError，由调用方决定 exit code
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import voxsplit  # 用 __file__ 找当前包的源码根

from voxsplit.core.constants import (
    INSTALLED_MARKER as MARKER,
    PYANNOTE_VERSION_SPEC,
    VENV_DIR,
    VENV_PYTHON,
)


class SetupError(RuntimeError):
    """lazy install / setup 过程的错误。"""


@dataclass
class VenvInfo:
    venv_path: Path
    venv_python: Path
    pyannote_version: str
    installed_marker: Path


def _voxsplit_source_root() -> Optional[Path]:
    """返回当前 voxsplit 包的源码根目录（含 pyproject.toml）。

    pipx 安装的 voxsplit 也会有源码（site-packages/voxsplit/...），但缺 pyproject。
    所以这里返回的"根"是 src/voxsplit 的祖父级；调用方据此判断能否 editable 安装。
    """
    pkg_dir = Path(voxsplit.__file__).resolve().parent  # .../src/voxsplit
    parent = pkg_dir.parent  # .../src
    grand = parent.parent
    if (grand / "pyproject.toml").is_file():
        return grand
    return None


def _read_marker() -> Optional[str]:
    if MARKER.is_file():
        return MARKER.read_text().strip()
    return None


def _write_marker(version: str) -> None:
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(f"pyannote.audio={version}\nspec={PYANNOTE_VERSION_SPEC}\n")


def _venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def _check_pyannote_version(py: Path) -> Optional[str]:
    """venv python 跑一行检查 pyannote.audio 版本。装上则返回字符串，否则 None。"""
    try:
        out = subprocess.run(
            [str(py), "-c", "import pyannote.audio; print(pyannote.audio.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _have_uv() -> str:
    """返回 uv 可执行路径；找不到抛 SetupError。"""
    p = shutil.which("uv")
    if not p:
        raise SetupError("未找到 uv（brew install uv）")
    return p


def _create_venv(uv_bin: str, *, verbose: bool) -> None:
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    cmd = [uv_bin, "venv", str(VENV_DIR), "--python", "3.12"]
    if verbose:
        print(f"[setup] uv venv {VENV_DIR}")
    proc = subprocess.run(cmd, capture_output=not verbose, text=True)
    if proc.returncode != 0:
        raise SetupError(f"uv venv 失败: {(proc.stderr or '').strip()[:500]}")


def _install_packages(uv_bin: str, py: Path, *, verbose: bool) -> None:
    """先装 pyannote.audio；再 editable 装 voxsplit 源码（让 worker 能 `-m`）。"""
    # 1) pyannote.audio + worker extras 已包含 torch/torchaudio
    cmd = [uv_bin, "pip", "install", "--python", str(py), PYANNOTE_VERSION_SPEC]
    if verbose:
        print(f"[setup] uv pip install pyannote.audio …（首次约 1-3 分钟）")
    proc = subprocess.run(cmd, capture_output=not verbose, text=True)
    if proc.returncode != 0:
        raise SetupError(f"uv pip install pyannote.audio 失败: {(proc.stderr or '').strip()[:500]}")

    # 2) editable 装 voxsplit（仅当源码可定位时；pipx 装的没源码 → 走 sys.path 注入兜底）
    src_root = _voxsplit_source_root()
    if src_root:
        cmd = [uv_bin, "pip", "install", "--python", str(py), "-e", str(src_root)]
        if verbose:
            print(f"[setup] uv pip install -e {src_root}")
        proc = subprocess.run(cmd, capture_output=not verbose, text=True)
        if proc.returncode != 0:
            raise SetupError(f"uv pip install -e voxsplit 失败: {(proc.stderr or '').strip()[:500]}")
    # 没有 src_root 时，diarize 子命令会通过 PYTHONPATH 注入主包路径（见 _ensure_voxsplit_importable）


def ensure_venv(*, verbose: bool = False) -> VenvInfo:
    """主入口：保证 venv 存在且 pyannote.audio 装好。

    Returns:
        VenvInfo
    """
    py = _venv_python(VENV_DIR)
    marker = _read_marker()
    cached_version = _check_pyannote_version(py) if py.is_file() else None

    if cached_version and marker and PYANNOTE_VERSION_SPEC in marker:
        if verbose:
            print(f"[setup] venv 已就绪（pyannote.audio={cached_version}）")
        return VenvInfo(
            venv_path=VENV_DIR,
            venv_python=py,
            pyannote_version=cached_version,
            installed_marker=MARKER,
        )

    uv_bin = _have_uv()
    if not VENV_DIR.is_dir():
        _create_venv(uv_bin, verbose=verbose)
    elif not py.is_file():
        # 残破的目录：清掉重建
        if verbose:
            print(f"[setup] venv 残破（{VENV_DIR}），重建…")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        _create_venv(uv_bin, verbose=verbose)

    _install_packages(uv_bin, py, verbose=verbose)

    version = _check_pyannote_version(py)
    if not version:
        raise SetupError("安装后 venv 仍 import pyannote.audio 失败")

    _write_marker(version)
    return VenvInfo(
        venv_path=VENV_DIR,
        venv_python=py,
        pyannote_version=version,
        installed_marker=MARKER,
    )


__all__ = ["ensure_venv", "SetupError", "VenvInfo", "VENV_DIR", "MARKER"]

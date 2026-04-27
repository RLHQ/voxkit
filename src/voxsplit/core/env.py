"""环境探测：torchcodec ffmpeg lib 路径修复。

torchcodec 在 macOS 上硬编码搜 /opt/homebrew/opt/ffmpeg/lib，但本机如果装的是
ffmpeg-full（lib 在 /opt/homebrew/lib/），就会找不到 libavutil。
通过 DYLD_LIBRARY_PATH 让 dlopen 多搜一处目录。

注意：DYLD_LIBRARY_PATH 必须在 import torch / pyannote 之前 export，否则
已经 dlopen 的库不会重试。所以这里只提供"应该 export 哪些"，由 lazy_install
spawn 子进程时通过 env= 传入。
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import List, Optional


# 候选 ffmpeg lib 目录（按优先级）
_MACOS_LIB_CANDIDATES = [
    "/opt/homebrew/lib",                # ffmpeg-full（Apple Silicon）
    "/usr/local/lib",                   # ffmpeg-full（Intel mac）
    "/opt/homebrew/opt/ffmpeg/lib",     # 标准 ffmpeg
]


def find_ffmpeg_lib_dir() -> Optional[str]:
    """返回包含 libavutil*.dylib 的第一个目录，找不到返回 None。

    macOS 专用；其他平台返回 None（torchcodec 在 Linux 上一般 apt 装的 ffmpeg 路径就对）。
    """
    if platform.system() != "Darwin":
        return None
    for cand in _MACOS_LIB_CANDIDATES:
        p = Path(cand)
        if not p.is_dir():
            continue
        # 检查 libavutil 任意版本是否存在
        if any(p.glob("libavutil*.dylib")):
            return str(p)
    return None


def patched_env(extra: Optional[dict] = None) -> dict:
    """构造 spawn 子进程时使用的环境，把 ffmpeg lib 目录 prepend 到 DYLD_LIBRARY_PATH。

    用法：
        env = patched_env()
        subprocess.run([...], env=env)
    """
    env = os.environ.copy()
    lib_dir = find_ffmpeg_lib_dir()
    if lib_dir:
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    if extra:
        env.update(extra)
    return env


def apply_in_process() -> Optional[str]:
    """在当前进程 export DYLD_LIBRARY_PATH（只对此后才 dlopen 的库生效）。

    返回 export 的目录；未找到返回 None。
    """
    lib_dir = find_ffmpeg_lib_dir()
    if not lib_dir:
        return None
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if lib_dir in existing.split(":"):
        return lib_dir
    os.environ["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
    return lib_dir


__all__ = ["find_ffmpeg_lib_dir", "patched_env", "apply_in_process"]

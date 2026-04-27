"""音频处理：视频抽音 + ffmpeg 版本检查。

策略：
- 输入扩展名是音频（wav/mp3/m4a/flac/aac/ogg/opus）→ 直接用，不抽
- 输入是视频（mp4/mov/mkv/webm/avi）→ ffmpeg 抽成 16kHz mono wav，写到临时目录
- ffmpeg major 版本不在 [4, 8] 区间则发 warn（不阻断）

ffmpeg 路径优先：PATH 找 → /opt/homebrew/bin/ffmpeg → /usr/local/bin/ffmpeg
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

_FFMPEG_PATH_CANDIDATES = [
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
]
_FFPROBE_PATH_CANDIDATES = [
    "/opt/homebrew/bin/ffprobe",
    "/usr/local/bin/ffprobe",
]


@dataclass
class AudioPrep:
    """音频准备结果。"""

    audio_path: Path                # 实际跑 diarization 的 wav
    duration_secs: float
    extracted_from: Optional[Path]  # 视频输入时记录原路径
    cleanup: Optional[Path]         # 需要在最后删除的临时文件（=抽出来的 wav）


def _find_executable(name: str, candidates: list[str]) -> Optional[str]:
    """先查 PATH，再回退到候选绝对路径列表。"""
    p = shutil.which(name)
    if p:
        return p
    for cand in candidates:
        if Path(cand).is_file():
            return cand
    return None


def find_ffmpeg() -> Optional[str]:
    return _find_executable("ffmpeg", _FFMPEG_PATH_CANDIDATES)


def find_ffprobe() -> Optional[str]:
    return _find_executable("ffprobe", _FFPROBE_PATH_CANDIDATES)


def get_ffmpeg_major_version() -> Optional[int]:
    """返回 ffmpeg major 版本号（如 8），找不到 ffmpeg 或解析失败返回 None。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    try:
        out = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    # 例如 "ffmpeg version 8.1 ..." / "ffmpeg version n4.4.4-..."
    m = re.search(r"ffmpeg version\s+n?(\d+)", out)
    return int(m.group(1)) if m else None


def probe_duration(path: Path) -> float:
    """ffprobe 拿媒体时长（秒）。"""
    ffprobe = find_ffprobe()
    if not ffprobe:
        raise RuntimeError("ffprobe 未找到，无法读取媒体时长")
    out = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    return float(out)


def prepare_audio(input_path: Path, *, tmp_root: Optional[Path] = None) -> AudioPrep:
    """根据输入扩展名决定是否抽音；返回 AudioPrep。

    - 音频输入：直接返回，cleanup=None
    - 视频输入：抽成 16kHz mono wav，cleanup 指向该 wav（调用方在最后 unlink）
    """
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    ext = input_path.suffix.lower()
    if ext in _AUDIO_EXTS:
        return AudioPrep(
            audio_path=input_path,
            duration_secs=probe_duration(input_path),
            extracted_from=None,
            cleanup=None,
        )
    if ext in _VIDEO_EXTS:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg 未找到，无法从视频抽音；建议 brew install ffmpeg-full")
        tmp_dir = Path(tmp_root) if tmp_root else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_wav = tmp_dir / f"voxsplit-{input_path.stem}-{input_path.stat().st_mtime_ns}.wav"
        # 16kHz mono PCM，pyannote 标准输入
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", str(input_path),
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(out_wav),
            ],
            check=True,
            timeout=600,
        )
        return AudioPrep(
            audio_path=out_wav,
            duration_secs=probe_duration(out_wav),
            extracted_from=input_path,
            cleanup=out_wav,
        )

    raise ValueError(
        f"不支持的扩展名 {ext}；支持：{sorted(_AUDIO_EXTS | _VIDEO_EXTS)}"
    )


__all__ = [
    "AudioPrep",
    "find_ffmpeg",
    "find_ffprobe",
    "get_ffmpeg_major_version",
    "probe_duration",
    "prepare_audio",
]

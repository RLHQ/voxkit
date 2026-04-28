"""音频处理：视频抽音 + ffmpeg 版本检查。

策略：
- 输入扩展名是音频（wav/mp3/m4a/flac/aac/ogg/opus）→ 直接用，不抽
- 输入是视频（mp4/mov/mkv/webm/avi）→ ffmpeg 抽成 16kHz mono wav，写到临时目录
- ffmpeg major 版本不在 [4, 8] 区间则发 warn（不阻断）

ffmpeg 路径优先：PATH 找 → /opt/homebrew/bin/ffmpeg → /usr/local/bin/ffmpeg
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
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


# 一次 CLI 调用里 ffmpeg/ffprobe 路径不会变；缓存避免每 chunk 重复扫 PATH。
@lru_cache(maxsize=1)
def find_ffmpeg() -> Optional[str]:
    return _find_executable("ffmpeg", _FFMPEG_PATH_CANDIDATES)


@lru_cache(maxsize=1)
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
        out_wav = tmp_dir / f"voxkit-{input_path.stem}-{input_path.stat().st_mtime_ns}.wav"
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


# ---------------------------------------------------------------------------
# Plan §A.4 — Remixr-aligned chunking
# ---------------------------------------------------------------------------

# 长音频分块阈值与参数（与 Remixr 对齐）
CHUNK_THRESHOLD_SECS = 900.0    # ≥15min 触发分块
CHUNK_DURATION_SECS = 600.0     # 单 chunk 10min
CHUNK_OVERLAP_SECS = 5.0        # 相邻 chunk overlap 5s


def chunk_thresholds_from_env() -> tuple[float, float, float]:
    """读取 chunk 切分参数，环境变量优先，缺省回退到模块常量。

    诊断 hatch：仅供 transcribe_pipeline 调用，便于 A/B 实验时强制改 chunk
    边界以复现"词被切成两半"类问题。生产 CLI 不暴露这些参数。

    Env vars：
      - ``VOXKIT_CHUNK_THRESHOLD_SECS``
      - ``VOXKIT_CHUNK_SECS``
      - ``VOXKIT_CHUNK_OVERLAP_SECS``

    Returns:
        ``(threshold_secs, chunk_secs, overlap_secs)``
    """
    threshold = float(
        os.environ.get("VOXKIT_CHUNK_THRESHOLD_SECS", CHUNK_THRESHOLD_SECS)
    )
    chunk = float(os.environ.get("VOXKIT_CHUNK_SECS", CHUNK_DURATION_SECS))
    overlap = float(
        os.environ.get("VOXKIT_CHUNK_OVERLAP_SECS", CHUNK_OVERLAP_SECS)
    )
    return threshold, chunk, overlap


@dataclass(frozen=True)
class ChunkSpec:
    """单个分块的位置与目标 wav 路径。"""

    index: int                  # 0-based
    start_secs: float           # 在原始音频中的偏移
    duration_secs: float        # 此 chunk 的实际时长
    out_wav: Path               # 此 chunk 写出的绝对路径


@dataclass(frozen=True)
class ChunkPlan:
    """整段音频的分块方案。"""

    chunks: list[ChunkSpec]     # 短音频时仅 1 个
    total_secs: float           # 原始音频总时长


def plan_chunks(
    duration_secs: float,
    work_dir: Path,
    *,
    threshold_secs: float = CHUNK_THRESHOLD_SECS,
    chunk_secs: float = CHUNK_DURATION_SECS,
    overlap_secs: float = CHUNK_OVERLAP_SECS,
) -> ChunkPlan:
    """纯计算的分块规划（无 I/O）。

    规则：
      - duration_secs <= threshold_secs → 单个 ChunkSpec 覆盖 [0, duration_secs]
      - 否则按 chunk_secs 切分，相邻 chunk overlap=overlap_secs；
        chunk i 起点 = i * (chunk_secs - overlap_secs)，时长为 chunk_secs，
        最后一个 chunk 时长被 clamp 到剩余音频长度。

    out_wav 路径形如：work_dir / "chunks" / "chunk_NNN.wav"。
    """
    chunks_dir = work_dir / "chunks"

    if duration_secs <= threshold_secs:
        spec = ChunkSpec(
            index=0,
            start_secs=0.0,
            duration_secs=float(duration_secs),
            out_wav=chunks_dir / "chunk_000.wav",
        )
        return ChunkPlan(chunks=[spec], total_secs=float(duration_secs))

    step = chunk_secs - overlap_secs
    if step <= 0:
        raise ValueError(
            f"overlap_secs ({overlap_secs}) 必须小于 chunk_secs ({chunk_secs})"
        )

    specs: list[ChunkSpec] = []
    i = 0
    while True:
        start = i * step
        if start >= duration_secs:
            break
        remaining = duration_secs - start
        dur = min(chunk_secs, remaining)
        specs.append(
            ChunkSpec(
                index=i,
                start_secs=float(start),
                duration_secs=float(dur),
                out_wav=chunks_dir / f"chunk_{i:03d}.wav",
            )
        )
        # 已经覆盖到末尾时停止（防止剩余 < step 时还多生成一个空 chunk）
        if start + dur >= duration_secs:
            break
        i += 1

    return ChunkPlan(chunks=specs, total_secs=float(duration_secs))


def normalize_to_wav_16k_mono(
    input_path: Path,
    out_wav: Path,
    *,
    ffmpeg_bin: Optional[Path] = None,
) -> None:
    """将任意输入归一化为 16kHz mono PCM wav。

    ffmpeg -y -i <input> -ar 16000 -ac 1 -f wav <out_wav>
    """
    from . import env as _env  # 局部 import 避免循环

    bin_path = str(ffmpeg_bin) if ffmpeg_bin else find_ffmpeg()
    if not bin_path:
        raise RuntimeError("ffmpeg 未找到，无法归一化音频")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        bin_path, "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(out_wav),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_env.patched_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg normalize 失败 (rc={proc.returncode}):\n{proc.stderr}"
        )


def extract_chunk(
    master_wav: Path,
    spec: ChunkSpec,
    *,
    ffmpeg_bin: Optional[Path] = None,
) -> None:
    """从 master_wav 切出 [start, start+duration] 写入 spec.out_wav。

    使用 input-side seek 提速：
        ffmpeg -y -ss <start> -t <dur> -i <master> -c:a pcm_s16le <out_wav>

    注：input seek 需要重新编码为 pcm_s16le，但因为 master_wav 已经是 16kHz mono PCM，
    成本极低。不使用 -c copy 是因为边界帧可能错位。
    """
    from . import env as _env

    bin_path = str(ffmpeg_bin) if ffmpeg_bin else find_ffmpeg()
    if not bin_path:
        raise RuntimeError("ffmpeg 未找到，无法切分 chunk")

    spec.out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        bin_path, "-y", "-loglevel", "error",
        "-ss", f"{spec.start_secs}",
        "-t", f"{spec.duration_secs}",
        "-i", str(master_wav),
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(spec.out_wav),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_env.patched_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg extract_chunk 失败 (chunk={spec.index}, rc={proc.returncode}):\n{proc.stderr}"
        )


__all__ = [
    "AudioPrep",
    "ChunkPlan",
    "ChunkSpec",
    "CHUNK_DURATION_SECS",
    "CHUNK_OVERLAP_SECS",
    "CHUNK_THRESHOLD_SECS",
    "chunk_thresholds_from_env",
    "extract_chunk",
    "find_ffmpeg",
    "find_ffprobe",
    "get_ffmpeg_major_version",
    "normalize_to_wav_16k_mono",
    "plan_chunks",
    "probe_duration",
    "prepare_audio",
]

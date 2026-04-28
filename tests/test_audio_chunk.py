"""分块规划 + ffmpeg 归一化/切片 测试（Plan §A.4）。

`plan_chunks` 是纯计算，覆盖各种边界。
`normalize_to_wav_16k_mono` 与 `extract_chunk` 需要真实 ffmpeg/ffprobe，
缺失时直接 skip。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voxkit.core.audio import (
    CHUNK_DURATION_SECS,
    CHUNK_OVERLAP_SECS,
    CHUNK_THRESHOLD_SECS,
    ChunkPlan,
    ChunkSpec,
    extract_chunk,
    find_ffmpeg,
    find_ffprobe,
    normalize_to_wav_16k_mono,
    plan_chunks,
)


# ---------------------------------------------------------------------------
# plan_chunks — 纯计算
# ---------------------------------------------------------------------------


def test_plan_chunks_short_audio_returns_single_chunk(tmp_path: Path) -> None:
    plan = plan_chunks(60.0, tmp_path)
    assert isinstance(plan, ChunkPlan)
    assert plan.total_secs == 60.0
    assert len(plan.chunks) == 1
    only = plan.chunks[0]
    assert only.index == 0
    assert only.start_secs == 0.0
    assert only.duration_secs == 60.0
    assert only.out_wav == tmp_path / "chunks" / "chunk_000.wav"


def test_plan_chunks_at_threshold_still_single(tmp_path: Path) -> None:
    """duration == threshold（900s）应仍然是单 chunk（边界包含）。"""
    plan = plan_chunks(CHUNK_THRESHOLD_SECS, tmp_path)
    assert len(plan.chunks) == 1
    assert plan.chunks[0].duration_secs == CHUNK_THRESHOLD_SECS


def test_plan_chunks_just_over_threshold_splits(tmp_path: Path) -> None:
    """超过阈值 1s 就应该分块。"""
    plan = plan_chunks(CHUNK_THRESHOLD_SECS + 1.0, tmp_path)
    assert len(plan.chunks) >= 2
    assert plan.chunks[0].start_secs == 0.0
    assert plan.chunks[0].duration_secs == CHUNK_DURATION_SECS


def test_plan_chunks_25min_three_chunks(tmp_path: Path) -> None:
    """1500s（25min）→ 3 chunks，starts=[0, 595, 1190], durs=[600, 600, 310]。"""
    plan = plan_chunks(1500.0, tmp_path)
    assert len(plan.chunks) == 3

    starts = [c.start_secs for c in plan.chunks]
    durs = [c.duration_secs for c in plan.chunks]
    assert starts == [0.0, 595.0, 1190.0]
    assert durs == [600.0, 600.0, 310.0]

    # 末尾应正好覆盖到 1500
    last = plan.chunks[-1]
    assert abs((last.start_secs + last.duration_secs) - 1500.0) < 1e-6


def test_plan_chunks_dryrun_video_3821s(tmp_path: Path) -> None:
    """voxsplit dryrun 视频时长 3821.6s → 验证覆盖范围与无重叠遗漏。"""
    duration = 3821.6
    plan = plan_chunks(duration, tmp_path)
    assert len(plan.chunks) >= 2

    # 第一块从 0 开始
    assert plan.chunks[0].start_secs == 0.0

    # 最后一块的 start + dur 应覆盖到 duration（容差 0.1s）
    last = plan.chunks[-1]
    assert abs((last.start_secs + last.duration_secs) - duration) < 0.1

    # 相邻 chunk 之间有 overlap_secs 重叠（除了最后一块的尾巴）
    step = CHUNK_DURATION_SECS - CHUNK_OVERLAP_SECS
    for i in range(1, len(plan.chunks)):
        assert plan.chunks[i].start_secs == pytest.approx(i * step)


def test_plan_chunks_very_short_input(tmp_path: Path) -> None:
    """0.5s 的脏输入也应当返回 1 chunk。"""
    plan = plan_chunks(0.5, tmp_path)
    assert len(plan.chunks) == 1
    assert plan.chunks[0].duration_secs == 0.5


def test_plan_chunks_paths_under_chunks_dir(tmp_path: Path) -> None:
    plan = plan_chunks(1500.0, tmp_path)
    chunks_dir = tmp_path / "chunks"
    for c in plan.chunks:
        assert c.out_wav.parent == chunks_dir
        # 文件名形如 chunk_000.wav, chunk_001.wav, ...
        assert c.out_wav.name == f"chunk_{c.index:03d}.wav"


def test_plan_chunks_invalid_overlap_raises(tmp_path: Path) -> None:
    """overlap >= chunk 是非法的。"""
    with pytest.raises(ValueError):
        plan_chunks(2000.0, tmp_path, chunk_secs=10.0, overlap_secs=10.0)


# ---------------------------------------------------------------------------
# normalize_to_wav_16k_mono / extract_chunk — 需要真实 ffmpeg
# ---------------------------------------------------------------------------


def _ffmpeg_or_skip() -> str:
    bin_path = find_ffmpeg()
    if not bin_path:
        pytest.skip("ffmpeg not available")
    return bin_path


def _ffprobe_or_skip() -> str:
    bin_path = find_ffprobe()
    if not bin_path:
        pytest.skip("ffprobe not available")
    return bin_path


def _generate_sine_wav(
    out: Path, duration_secs: float, *, sample_rate: int = 44100, channels: int = 2
) -> None:
    """生成 sine 测试音频。"""
    ffmpeg = _ffmpeg_or_skip()
    out.parent.mkdir(parents=True, exist_ok=True)
    layout = "stereo" if channels == 2 else "mono"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_secs}:sample_rate={sample_rate}",
        "-ac", str(channels),
        "-channel_layout", layout,
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_stream_format(p: Path) -> dict:
    """读取 sample_rate, channels, format duration。"""
    ffprobe = _ffprobe_or_skip()
    out = subprocess.check_output(
        [
            ffprobe, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=0",
            str(p),
        ],
        text=True,
    ).strip()
    info: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def test_normalize_to_wav_16k_mono_real_ffmpeg(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    _ffprobe_or_skip()

    src = tmp_path / "src.wav"
    _generate_sine_wav(src, duration_secs=25.0, sample_rate=44100, channels=2)
    assert src.is_file()

    out = tmp_path / "16k.wav"
    normalize_to_wav_16k_mono(src, out)
    assert out.is_file()
    assert out.stat().st_size > 0

    info = _probe_stream_format(out)
    assert int(info["sample_rate"]) == 16000
    assert int(info["channels"]) == 1


def test_normalize_to_wav_creates_parent_dirs(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    src = tmp_path / "src.wav"
    _generate_sine_wav(src, duration_secs=2.0)

    out = tmp_path / "deep" / "nested" / "out.wav"
    normalize_to_wav_16k_mono(src, out)
    assert out.is_file()


def test_plan_chunks_for_short_real_audio(tmp_path: Path) -> None:
    """整合：25s 的音频应当只产生 1 个 chunk（远低于 900s 阈值）。"""
    plan = plan_chunks(25.0, tmp_path)
    assert len(plan.chunks) == 1


def test_plan_chunks_synthetic_long_audio_three_chunks(tmp_path: Path) -> None:
    """1500s 不实际生成（太大），仅校验 planner 数学。"""
    plan = plan_chunks(1500.0, tmp_path)
    assert len(plan.chunks) == 3


def test_extract_chunk_real_ffmpeg(tmp_path: Path) -> None:
    """生成 25s 母 wav，切出 10s..15s 一段，验证时长 ≈5s。"""
    _ffmpeg_or_skip()
    _ffprobe_or_skip()

    # 先归一化到 16k mono（模拟真实 pipeline 的 master_wav）
    raw = tmp_path / "raw.wav"
    _generate_sine_wav(raw, duration_secs=25.0, sample_rate=44100, channels=2)
    master = tmp_path / "master_16k.wav"
    normalize_to_wav_16k_mono(raw, master)

    spec = ChunkSpec(
        index=0,
        start_secs=10.0,
        duration_secs=5.0,
        out_wav=tmp_path / "chunks" / "chunk_000.wav",
    )
    extract_chunk(master, spec)

    assert spec.out_wav.is_file()
    info = _probe_stream_format(spec.out_wav)
    assert int(info["sample_rate"]) == 16000
    assert int(info["channels"]) == 1
    # ffmpeg input-side seek 精度通常 < 0.1s
    assert abs(float(info["duration"]) - 5.0) < 0.1


def test_extract_chunk_creates_parent_dirs(tmp_path: Path) -> None:
    _ffmpeg_or_skip()
    raw = tmp_path / "raw.wav"
    _generate_sine_wav(raw, duration_secs=5.0)
    master = tmp_path / "master.wav"
    normalize_to_wav_16k_mono(raw, master)

    spec = ChunkSpec(
        index=0,
        start_secs=1.0,
        duration_secs=2.0,
        out_wav=tmp_path / "deep" / "nested" / "chunks" / "chunk_000.wav",
    )
    extract_chunk(master, spec)
    assert spec.out_wav.is_file()


def test_extract_chunk_failure_raises(tmp_path: Path) -> None:
    """master 不存在时应抛 RuntimeError 带 stderr。"""
    _ffmpeg_or_skip()
    spec = ChunkSpec(
        index=0,
        start_secs=0.0,
        duration_secs=1.0,
        out_wav=tmp_path / "out.wav",
    )
    with pytest.raises(RuntimeError, match="extract_chunk"):
        extract_chunk(tmp_path / "does_not_exist.wav", spec)

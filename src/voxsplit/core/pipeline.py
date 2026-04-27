"""pyannote 推理 worker：脚本入口被 lazy venv 的 python 子进程调用。

为什么是独立 worker？
- voxsplit CLI 主进程零 pyannote 依赖（pipx 装即可秒启）
- pyannote / torch / torchaudio 走独立 venv (~/.local/share/voxsplit/venv/)
- spawn `<venv>/bin/python -m voxsplit.core.pipeline ...` 跑这个文件

输入：CLI args（音频路径 / model / device / num_speakers ...）
输出：把 DiarizationOutput JSON 写到 stdout（最后一行）；进度走 stderr NDJSON

注意：本模块在主进程 import 不会 fail（只 import stdlib + voxsplit.io），
torch / pyannote 仅在 main() 内 lazy import。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# 主进程也可能 import 本模块（用于命令路由），保持顶部 import 干净
from voxsplit import SCHEMA_VERSION
from voxsplit.core.constants import (
    DEFAULT_MODEL,
    ExitCode,
    MODEL_ALIASES,
    WORKER_JSON_SENTINEL,
)
from voxsplit.io.progress import ProgressEmitter
from voxsplit.io.schema import (
    AudioInfo,
    DiarizationOutput,
    Segment,
    SpeakerInfo,
)


def _resolve_model_id(short: str) -> str:
    try:
        return MODEL_ALIASES[short]
    except KeyError:
        raise ValueError(f"未知 model: {short}（合法：{list(MODEL_ALIASES)}）") from None


def _annotation_from_result(result):
    """pyannote 4.x → result.exclusive_speaker_diarization；3.x → result 本身。"""
    if hasattr(result, "exclusive_speaker_diarization"):
        return result.exclusive_speaker_diarization
    if hasattr(result, "itertracks"):
        return result
    raise RuntimeError(
        f"未识别的 pyannote 输出类型: {type(result).__name__}"
        f"（attrs={[x for x in dir(result) if not x.startswith('_')]}）"
    )


def _rank_speakers(raw_segments: List[dict], labels_mode: str) -> tuple[List[SpeakerInfo], dict[str, str]]:
    """统计每个 raw speaker 的总时长，按降序映射成 Speaker 1/2/3...

    labels_mode ∈ {ranked, raw}：
      - ranked: 返回映射 SPEAKER_01 → "Speaker 1"
      - raw:    映射保持 raw（id == raw_id）
    """
    durations: dict[str, float] = {}
    for s in raw_segments:
        durations[s["raw_speaker"]] = durations.get(s["raw_speaker"], 0.0) + (s["end"] - s["start"])

    sorted_pairs = sorted(durations.items(), key=lambda kv: -kv[1])
    if labels_mode == "ranked":
        mapping = {raw: f"Speaker {i + 1}" for i, (raw, _) in enumerate(sorted_pairs)}
    else:
        mapping = {raw: raw for raw, _ in sorted_pairs}

    speakers = [
        SpeakerInfo(
            id=mapping[raw],
            raw_id=raw,
            total_duration_secs=round(dur, 3),
        )
        for raw, dur in sorted_pairs
    ]
    return speakers, mapping


def run_diarize(
    *,
    audio_path: Path,
    audio_duration_secs: float,
    extracted_from: Optional[Path],
    model: str,
    device_pref: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    labels_mode: str,
    progress: ProgressEmitter,
) -> DiarizationOutput:
    """核心调用：加载 pipeline → 跑 diarization → 组装 DiarizationOutput。

    全部 torch / pyannote 相关 import 都在函数内（保持模块顶部干净）。
    """
    progress.progress("model_load", 0)
    from pyannote.audio import Pipeline  # type: ignore

    from voxsplit.core.device import select_device

    model_id = _resolve_model_id(model)
    pipeline = Pipeline.from_pretrained(model_id)
    if pipeline is None:
        raise RuntimeError(
            "pyannote pipeline 加载返回 None；常见原因："
            f"\n  1) 未在 https://huggingface.co/{model_id} 点 Accept"
            "\n  2) HF token 缺失或无效（~/.cache/huggingface/token）"
            "\n  3) 网络问题导致首次模型下载失败"
        )

    device, device_name = select_device(device_pref)
    pipeline.to(device)
    progress.progress("model_load", 100)
    progress.info(f"device={device_name} model={model_id}")

    diar_kwargs: dict = {}
    if num_speakers is not None:
        diar_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diar_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diar_kwargs["max_speakers"] = max_speakers

    progress.progress("diarize", 0)
    t0 = time.time()
    result = pipeline(str(audio_path), **diar_kwargs)
    elapsed = time.time() - t0
    progress.progress("diarize", 100)

    annotation = _annotation_from_result(result)

    raw_segments: List[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_segments.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "raw_speaker": speaker,
            }
        )

    speakers, mapping = _rank_speakers(raw_segments, labels_mode)

    segments = [
        Segment(
            start=s["start"],
            end=s["end"],
            speaker=mapping[s["raw_speaker"]],
            raw_speaker=s["raw_speaker"],
        )
        for s in raw_segments
    ]

    rtf = elapsed / audio_duration_secs if audio_duration_secs > 0 else 0.0

    return DiarizationOutput(
        schema_version=SCHEMA_VERSION,
        audio=AudioInfo(
            path=str(audio_path),
            duration_secs=round(audio_duration_secs, 3),
            extracted_from=str(extracted_from) if extracted_from else None,
        ),
        device=device_name,
        model=model_id,
        rtf=round(rtf, 4),
        elapsed_secs=round(elapsed, 2),
        num_speakers=len(speakers),
        speakers=speakers,
        segments=segments,
        warnings=[],
    )


# ─────────────────────────────────────────────────────────────────
# Worker CLI 入口（被 spawn 的 venv python 调用）
# ─────────────────────────────────────────────────────────────────

def _build_worker_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxsplit-pipeline-worker")
    p.add_argument("--audio", required=True)
    p.add_argument("--audio-duration-secs", type=float, required=True)
    p.add_argument("--extracted-from", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-speakers", type=int, default=None)
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--speaker-labels", choices=["ranked", "raw"], default="ranked")
    p.add_argument("--json-events", action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_worker_parser().parse_args(argv)
    progress = ProgressEmitter(json_events=args.json_events)

    try:
        out = run_diarize(
            audio_path=Path(args.audio),
            audio_duration_secs=args.audio_duration_secs,
            extracted_from=Path(args.extracted_from) if args.extracted_from else None,
            model=args.model,
            device_pref=args.device,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            labels_mode=args.speaker_labels,
            progress=progress,
        )
    except KeyboardInterrupt:
        progress.error("INTERRUPTED", "用户中断")
        return int(ExitCode.INTERRUPTED)
    except Exception as e:
        progress.error("WORKER_FAILED", str(e))
        return int(ExitCode.WORKER_FAILED)

    # 加 sentinel 前缀避免 torch / pyannote 偶然的 print 污染主进程的 JSON 解析
    sys.stdout.write(WORKER_JSON_SENTINEL + out.model_dump_json(by_alias=True) + "\n")
    sys.stdout.flush()
    progress.done(elapsed_secs=out.elapsed_secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

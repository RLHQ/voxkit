"""``voxkit reseg`` 子命令：proofread 后做二次语义切分。

设计动机（详见 docs/eval-baseline-observations.md §8）：

whisper.cpp 中文 ASR 输出无标点，第一 pass `_build_cjk_atoms` 没有 medium
punctuation 可切，cue 切分密度上限是 `_pack_cjk_atom_block` 的 soft_max
（~18 char）。proofread 阶段 LLM 加完标点后，`correctedText` 里出现
「，。？！；、」承载了约 80% 气口边界——本命令把这些 corrected cue 当
带标点 ASR segment 喂回 reseg，让 atom 切分利用 proofread 的标点。

小宁子 10min fixture 实测（v0.7.0 0.6.0 transcribe + proofread → reseg）：

- 单 pass cue=200, precision=0.901, recall=0.559, F1=0.690
- 双 pass cue=210, precision=**0.906**, recall=**0.597**, F1=**0.720**

precision 不退反升，recall +0.038，F1 +0.030。前提是 input cue 已足够细
（avg < 3s），否则 `_estimate_char_time` 线性插值在长 cue 内会丢失精度。

零 LLM 零网络（不像 proofread），可在 CI 频繁跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from voxkit.core.semantic_resegment import (
    SubtitleCue,
    resegment_for_subtitles,
)
from voxkit.io.schema import RemixrSegment
from voxkit.io.srt import to_subtitles_srt_from_cues


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "reseg",
        help=(
            "把 proofread 加完标点的 cue 喂回 semantic_resegment，输出 "
            "subtitles.cues.reseg2.json。零 LLM 零网络。详见 docs/eval-baseline-observations.md §8"
        ),
    )
    p.add_argument(
        "workdir",
        help="voxkit transcribe + proofread 的 workdir（需含 subtitles.proofread.json）",
    )
    p.add_argument(
        "--language",
        default=None,
        help="语种代码；缺省继承 proofread.json 的 language 字段",
    )
    p.add_argument(
        "--emit-srt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="同步渲染 subtitles.reseg2.srt（不覆盖原 subtitles.srt）",
    )
    p.add_argument(
        "--speaker-prefix",
        choices=["auto", "always", "never"],
        default="auto",
        dest="speaker_prefix",
        help=(
            "reseg2.srt 每条 cue 是否加 'Speaker X: ' 前缀。"
            "auto = 仅在 ≥2 个 informative speaker 时加（默认；占位符不计入）；"
            "always = v0.7.1 之前的旧行为；never = 永不加。"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的 subtitles.cues.reseg2.json",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    if not workdir.is_dir():
        sys.stderr.write(f"error: workdir not a directory: {workdir}\n")
        return 2

    proof_path = workdir / "subtitles.proofread.json"
    if not proof_path.is_file():
        sys.stderr.write(
            f"error: subtitles.proofread.json missing in {workdir}; "
            "run `voxkit proofread` first\n"
        )
        return 2

    out_path = workdir / "subtitles.cues.reseg2.json"
    if out_path.is_file() and not args.force:
        sys.stderr.write(
            f"error: {out_path.name} exists; use --force to overwrite\n"
        )
        return 2

    doc: dict[str, Any] = json.loads(proof_path.read_text(encoding="utf-8"))
    language = args.language or doc.get("language") or "auto"

    # 把 proofread cue 包装成 RemixrSegment（CJK 路径不需要 word timestamps）
    segments = [
        RemixrSegment(
            id=c.get("cueId") or f"cue_{i:06d}",
            speaker=c.get("speaker") or "Speaker A",
            start=float(c.get("sourceStart", 0.0)),
            end=float(c.get("sourceEnd", 0.0)),
            text=c.get("correctedText", "") or "",
            words=[],
        )
        for i, c in enumerate(doc.get("cues") or [], 1)
    ]

    new_cues: list[SubtitleCue] = resegment_for_subtitles(
        segments, language=language
    )

    out_doc = {
        "schemaVersion": 2,
        "sourceId": doc.get("sourceId"),
        "language": language,
        "params": {
            "basedOn": "subtitles.proofread.json",
            "inputCueCount": len(segments),
            "outputCueCount": len(new_cues),
        },
        "cues": [
            {
                "cueId": f"cue_{i:06d}",
                "start": c.start,
                "end": c.end,
                "text": c.text,
                "speaker": c.speaker,
            }
            for i, c in enumerate(new_cues, 1)
        ],
    }
    out_path.write_text(
        json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.emit_srt:
        srt_path = workdir / "subtitles.reseg2.srt"
        srt_path.write_text(
            to_subtitles_srt_from_cues(
                new_cues, speaker_prefix=getattr(args, "speaker_prefix", "auto")
            ),
            encoding="utf-8",
        )

    sys.stdout.write(
        f"reseg done: {len(segments)} proofread cues → {len(new_cues)} reseg2 cues\n"
        f"  → {out_path}\n"
        + (f"  → {workdir / 'subtitles.reseg2.srt'}\n" if args.emit_srt else "")
    )
    return 0

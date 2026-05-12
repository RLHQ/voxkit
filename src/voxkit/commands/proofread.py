"""``voxkit proofread`` 子命令入口。

只负责 argparse → ``ProofreadRequest`` 的转换，业务在
:mod:`voxkit.core.proofread_pipeline` 里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from voxkit.core.proofread_pipeline import ProofreadRequest, run_proofread
from voxkit.llm.errors import LLMError


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "proofread",
        help=(
            "对 subtitles.cues.json 跑 LLM 校对，产出 subtitles.proofread.json "
            "(state=draft)。需要环境变量 DEEPSEEK_API_KEY 或对应 provider 的 key"
        ),
    )
    p.add_argument("workdir", help="voxkit transcribe 的 workdir（含 subtitles.cues.json）")
    p.add_argument("--provider", default="deepseek", help="LLM provider 名（默认 deepseek）")
    p.add_argument("--model", default=None, help="覆盖 provider 默认 model")
    p.add_argument(
        "--language",
        default=None,
        help="校对语言代码；缺省继承 cues.params.language，再缺省即 auto",
    )
    p.add_argument(
        "--edit-level",
        choices=["punctuation", "light", "standard", "strict"],
        default="standard",
        help="编辑强度（默认 standard）",
    )
    p.add_argument("--glossary", default=None, help="可选 glossary.json 路径")
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=6000,
        help="单 batch 输入 token 上限（默认 6000；保守估算 CJK 0.5、Latin 0.25 token/char）",
    )
    p.add_argument(
        "--max-cues-per-batch",
        type=int,
        default=40,
        help="单 batch cue 数上限（默认 40）",
    )
    p.add_argument("--context-prev", type=int, default=8, help="batch 上文 cue 数（默认 8）")
    p.add_argument("--context-next", type=int, default=4, help="batch 下文 cue 数（默认 4）")
    # --force 三档（语义见 ProofreadRequest.force_level）：
    #   --force          → 只覆盖 draft（默认安全档）
    #   --force-reviewed → 也覆盖 reviewed（确认要丢人工 confirm 的产物）
    #   --force-final    → 也覆盖 final（确认要丢人工锁定的产物）
    p.add_argument(
        "--force",
        action="store_true",
        help="覆盖 draft 状态的 subtitles.proofread.json 并清空 work/proofread/；遇 reviewed/final 仍会拒绝",
    )
    p.add_argument(
        "--force-reviewed",
        action="store_true",
        help="允许覆盖 reviewed 状态（隐含 --force）。注意会丢失人工 confirm 元数据",
    )
    p.add_argument(
        "--force-final",
        action="store_true",
        help="允许覆盖 final 状态（隐含 --force-reviewed）。**销毁人工 lock 产物，慎用**",
    )
    p.add_argument(
        "--json-events",
        action="store_true",
        help="stderr 改为 NDJSON 事件协议（机器消费）",
    )
    p.add_argument("--timeout", type=float, default=60.0, help="单次 LLM 请求超时（秒）")


def _resolve_force_level(args: argparse.Namespace) -> str | None:
    """三档 force 旗标 → ``ForceLevel`` 字符串。任一高级位隐含低级。"""
    if getattr(args, "force_final", False):
        return "final"
    if getattr(args, "force_reviewed", False):
        return "reviewed"
    if args.force:
        return "draft"
    return None


def run(args: argparse.Namespace) -> int:
    req = ProofreadRequest(
        workdir=Path(args.workdir),
        provider=args.provider,
        model=args.model,
        language=args.language,
        edit_level=args.edit_level,
        glossary_path=Path(args.glossary) if args.glossary else None,
        max_input_tokens=args.max_input_tokens,
        max_cues_per_batch=args.max_cues_per_batch,
        context_prev=args.context_prev,
        context_next=args.context_next,
        force_level=_resolve_force_level(args),
        json_events=args.json_events,
        timeout_s=args.timeout,
    )
    try:
        summary: dict[str, Any] = run_proofread(req)
    except FileNotFoundError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    except FileExistsError as e:
        sys.stderr.write(f"error: {e}\n")
        return 3
    except LLMError as e:
        sys.stderr.write(f"LLM error: {e}\n")
        return 4

    if not args.json_events:
        sys.stderr.write(
            f"proofread done: cues={summary.get('changedCueRate', 0):.0%} changed, "
            f"{summary.get('reviewCueRate', 0):.0%} need review, "
            f"{summary.get('promptTokens', 0)} + {summary.get('completionTokens', 0)} tokens\n"
        )
    return 0

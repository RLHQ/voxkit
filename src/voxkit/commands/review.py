"""``voxkit review`` 子命令入口。

提供两个动词：

  - ``voxkit review confirm <artifact_path> --reviewer "name"``：``draft → reviewed``
  - ``voxkit review lock <artifact_path>``：``reviewed → final``

业务全部委派给 :mod:`voxkit.core.lifecycle`；本文件只做 argparse → 调用 → 错误打印。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voxkit.core.lifecycle import (
    LifecycleError,
    detect_artifact_kind,
    mirror_to_manifest,
    transition_state,
)


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "review",
        help=(
            "推进 subtitles.proofread.json / subtitles.<lang>.json 的 state："
            "confirm (draft→reviewed) / lock (reviewed→final)"
        ),
    )
    verbs = p.add_subparsers(dest="verb", required=True, metavar="VERB")

    # confirm: draft → reviewed
    pc = verbs.add_parser(
        "confirm",
        help="把 artifact 从 draft 推进到 reviewed，写入 reviewedBy/reviewedAt",
    )
    pc.add_argument(
        "artifact_path",
        help="待 confirm 的 artifact 路径（subtitles.proofread.json 或 subtitles.<lang>.json）",
    )
    pc.add_argument(
        "--reviewer",
        required=True,
        help="评审人姓名（写入 reviewedBy）",
    )
    pc.add_argument(
        "--workdir",
        default=None,
        help="manifest 镜像目标 workdir；默认取 artifact 文件所在目录",
    )

    # lock: reviewed → final
    pl = verbs.add_parser(
        "lock",
        help="把 artifact 从 reviewed 推进到 final（锁定发布）",
    )
    pl.add_argument(
        "artifact_path",
        help="待 lock 的 artifact 路径",
    )
    pl.add_argument(
        "--workdir",
        default=None,
        help="manifest 镜像目标 workdir；默认取 artifact 文件所在目录",
    )


def _resolve_workdir(args: argparse.Namespace, artifact_path: Path) -> Path:
    if getattr(args, "workdir", None):
        return Path(args.workdir)
    return artifact_path.parent


def run(args: argparse.Namespace) -> int:
    verb: str = args.verb
    artifact_path = Path(args.artifact_path)
    workdir = _resolve_workdir(args, artifact_path)

    try:
        if verb == "confirm":
            updated = transition_state(
                artifact_path, to="reviewed", reviewer=args.reviewer
            )
        elif verb == "lock":
            updated = transition_state(artifact_path, to="final")
        else:  # pragma: no cover — argparse ensures one of the above
            sys.stderr.write(f"error: unknown verb {verb!r}\n")
            return 2

        # manifest 镜像（失败也只打 warning，不污染 artifact 写入）
        try:
            mirror_to_manifest(workdir, artifact_path, updated)
        except LifecycleError as exc:
            sys.stderr.write(f"warning: manifest mirror failed: {exc}\n")

        kind = detect_artifact_kind(artifact_path)
        state = updated.get("state")
        reviewer = updated.get("reviewedBy")
        bits = [f"state={state}"]
        if reviewer:
            bits.append(f"reviewer={reviewer}")
        sys.stdout.write(
            f"{verb} {artifact_path.name} ({kind}): {', '.join(bits)}\n"
        )
        return 0
    except LifecycleError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except FileNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

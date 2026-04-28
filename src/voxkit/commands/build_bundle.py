"""voxkit build-bundle — 把本机 HF cache 中 voxkit 用到的 4 个模型打包。

输出：
- ``voxkit-models.tar.gz`` —— 主 bundle，含 4 个 ``models--pyannote--*`` 目录
  + ``ATTRIBUTION.md``（CC-BY-4.0 强制要求）
- ``voxkit-models.manifest.json`` —— 独立 release asset，含每文件 sha256 +
  bundle 整体 sha256，供 fetch-bundle 校验

Bundle 内部结构（保留 HF cache 标准）：
    voxkit-models.tar.gz
    ├── ATTRIBUTION.md
    └── hub/
        ├── models--pyannote--speaker-diarization-3.1/
        │   ├── blobs/<sha>
        │   ├── snapshots/<commit>/config.yaml -> ../../blobs/<sha>
        │   └── refs/main
        ├── models--pyannote--segmentation-3.0/...
        ├── models--pyannote--speaker-diarization-community-1/...
        └── models--pyannote--wespeaker-voxceleb-resnet34-LM/...

Bundle 内部不带 ``hub/`` 包装，因为用户 HF cache 里通常还有别的模型/数据集
（whisper / transformers / datasets ...）—— 必须**严格只打包 voxkit 必需的
4 个 ``models--pyannote--*`` 目录**，避免误打包数十 GB 无关数据。

实际结构：
    voxkit-models.tar.gz
    ├── ATTRIBUTION.md
    ├── models--pyannote--speaker-diarization-3.1/...
    ├── models--pyannote--segmentation-3.0/...
    ├── models--pyannote--speaker-diarization-community-1/...
    └── models--pyannote--wespeaker-voxceleb-resnet34-LM/...

fetch-bundle 直接 ``tar -x -C ~/.cache/huggingface/hub`` merge 进 hub。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import voxkit
from voxkit.core.bundle import (
    AuxFileEntry,
    BundleManifest,
    FileEntry,
    ModelEntry,
    hf_hub_cache_dir,
    render_attribution,
    repo_id_to_dirname,
    sha256_file,
)
from voxkit.core.constants import (
    BUNDLE_AUX_FILES,
    BUNDLE_FILENAME,
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_MODELS,
    ExitCode,
)


# ── HF cache 扫描 ─────────────────────────────────────────────────────────
def _scan_model_dir(model_dir: Path, repo_id: str, license_name: str) -> ModelEntry:
    """扫描 ``models--xxx--yyy/`` 目录，构造 ModelEntry。

    要求结构：
      - ``refs/main`` 存在且非空（commit hash）
      - ``snapshots/<commit>/`` 至少 1 个文件
      - 所有 snapshot 文件实际指向 ``blobs/<sha>``（symlink）

    返回的 FileEntry 是 ``hub/`` 视角下的相对路径，不含 ``hub/`` 前缀本身。
    """
    refs_main = model_dir / "refs" / "main"
    if not refs_main.is_file():
        raise RuntimeError(f"{model_dir.name}: 缺少 refs/main")
    commit = refs_main.read_text().strip()
    if not commit:
        raise RuntimeError(f"{model_dir.name}: refs/main 为空")

    snap_dir = model_dir / "snapshots" / commit
    if not snap_dir.is_dir():
        raise RuntimeError(f"{model_dir.name}: snapshots/{commit} 不存在")

    files: List[FileEntry] = []
    total_size = 0

    for fp in sorted(model_dir.rglob("*")):
        if fp.is_dir():
            continue
        rel = fp.relative_to(model_dir.parent)  # 相对于 hub/

        if fp.is_symlink():
            target = os.readlink(fp)
            # 校验 symlink 指向的实际 blob（必须在 model_dir 内可解析）
            resolved = (fp.parent / target).resolve()
            if not resolved.is_file():
                raise RuntimeError(f"{rel}: symlink 指向不存在 ({target})")
            files.append(FileEntry(
                path=str(rel), size=resolved.stat().st_size,
                sha256=sha256_file(resolved),
                is_symlink=True, symlink_target=target,
            ))
        else:
            sz = fp.stat().st_size
            total_size += sz
            files.append(FileEntry(
                path=str(rel), size=sz, sha256=sha256_file(fp),
                is_symlink=False,
            ))

    return ModelEntry(
        repo_id=repo_id,
        commit=commit,
        license=license_name,
        source_url=f"https://huggingface.co/{repo_id}",
        total_size=total_size,
        files=files,
    )


# ── aux 文件探测 ─────────────────────────────────────────────────────────
def _resolve_aux_source(spec: dict) -> Optional[Path]:
    """按 ``source_candidates`` 顺序找第一个存在的 aux 文件源。找不到返回 None。"""
    for cand in spec.get("source_candidates", []):
        p = Path(cand).expanduser()
        if p.is_file():
            return p
    return None


def _scan_aux_files(specs: List[dict]) -> List[tuple[dict, Path, AuxFileEntry]]:
    """扫描 aux 文件清单，返回 (spec, source_path, AuxFileEntry) 三元组列表。

    缺失的 aux 文件不抛错，只发 stderr 警告并跳过——这样开发者机器上没装
    whisper-cpp 也能继续构建模型 bundle。
    """
    out: List[tuple[dict, Path, AuxFileEntry]] = []
    for spec in specs:
        src = _resolve_aux_source(spec)
        if src is None:
            print(
                f"⚠️  aux file {spec['filename']} not found locally; "
                f"bundle will not include {spec['name']}",
                file=sys.stderr,
            )
            continue
        size = src.stat().st_size
        sha = sha256_file(src)
        # target_path 在 manifest 里**保持 spec 原样**（包含 ~），由 fetch 端按当前
        # 用户机器 expanduser。绝不要在 build 端展开成 /Users/<builder>/…——那会把
        # builder 的用户名烤进 manifest，让其他用户/机器拉到 release 后写到错误路径。
        target = spec["target"]
        entry = AuxFileEntry(
            name=spec["name"],
            filename=spec["filename"],
            license=spec["license"],
            sha256=sha,
            size_bytes=size,
            target_path=target,
        )
        print(f"  ✅ aux {spec['name']}: {src} ({size/1024:.1f} KB)")
        out.append((spec, src, entry))
    return out


# ── tar 打包（保留 symlinks） ────────────────────────────────────────────
def _tar_create(
    bundle_path: Path,
    hub_root: Path,
    model_dirnames: List[str],
    attribution_text: str,
    aux_sources: List[tuple[dict, Path, AuxFileEntry]] | None = None,
) -> None:
    """打包指定的若干 ``models--xxx--yyy/`` 目录到 bundle.tar.gz。

    严格只打包传入的 ``model_dirnames``——hub_root 下其他模型/数据集
    （whisper/transformers/datasets/...）不会被纳入。

    内部 ``snapshots/<commit>/file -> ../../blobs/<sha>`` 这种 symlink 自然保留
    （不用 ``tar -h``，那会把它们 dereference 成普通文件）。

    aux 文件被复制到 staging/aux/<filename> 后跟模型目录一起入 tar，bundle 内
    的相对路径是 ``aux/<filename>``。
    """
    if not model_dirnames:
        raise ValueError("model_dirnames 为空，没有任何模型可打包")

    with tempfile.TemporaryDirectory(prefix="voxkit-bundle-") as tmpd:
        staging = Path(tmpd)
        (staging / "ATTRIBUTION.md").write_text(attribution_text)

        staging_members = ["ATTRIBUTION.md"]
        if aux_sources:
            aux_dir = staging / "aux"
            aux_dir.mkdir()
            for _spec, src, entry in aux_sources:
                shutil.copyfile(src, aux_dir / entry.filename)
            staging_members.append("aux")

        cmd = [
            "tar", "-c", "-z", "-f", str(bundle_path),
            "-C", str(staging), *staging_members,
            "-C", str(hub_root), *model_dirnames,
        ]
        subprocess.run(cmd, check=True)


# ── 主流程 ─────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> int:
    hub = Path(args.hf_cache).expanduser().resolve() if args.hf_cache else hf_hub_cache_dir()
    if not hub.is_dir():
        print(f"❌ HF cache 目录不存在: {hub}", file=sys.stderr)
        return int(ExitCode.GENERIC_FAIL)

    bundle_version = args.bundle_version
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / BUNDLE_FILENAME
    manifest_path = out_dir / BUNDLE_MANIFEST_FILENAME

    print(f"[build] HF cache: {hub}")
    print(f"[build] 扫描 {len(BUNDLE_MODELS)} 个模型 ...")

    models: List[ModelEntry] = []
    for spec in BUNDLE_MODELS:
        repo_id = spec["repo_id"]
        license_name = spec["license"]
        model_dir = hub / repo_id_to_dirname(repo_id)
        if not model_dir.is_dir():
            print(f"❌ 缺少模型目录: {model_dir}", file=sys.stderr)
            print(f"   提示：先跑一次 voxkit diarize 让模型下载到 HF cache", file=sys.stderr)
            return int(ExitCode.GENERIC_FAIL)
        try:
            entry = _scan_model_dir(model_dir, repo_id, license_name)
        except RuntimeError as e:
            print(f"❌ {e}", file=sys.stderr)
            return int(ExitCode.GENERIC_FAIL)
        print(f"  ✅ {repo_id}: commit={entry.commit[:12]}  files={len(entry.files)}  {entry.total_size/1024/1024:.1f} MB")
        models.append(entry)

    # ── 扫 aux 文件（缺失只警告不报错）────────────────────────────
    print(f"[build] 扫描 {len(BUNDLE_AUX_FILES)} 个 aux 文件 ...")
    aux_sources = _scan_aux_files(list(BUNDLE_AUX_FILES))
    aux_entries = [e for (_s, _p, e) in aux_sources]

    # 先写 manifest 占位（attribution 模板需要 manifest 数据），bundle sha256 等打包后回填
    manifest = BundleManifest(
        schema_version="1",
        voxkit_version=voxkit.__version__,
        bundle_version=bundle_version,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        bundle_filename=BUNDLE_FILENAME,
        bundle_sha256="",
        bundle_size=0,
        models=models,
        aux_files=aux_entries,
    )

    print(f"[build] 打包 → {bundle_path}")
    model_dirnames = [repo_id_to_dirname(spec["repo_id"]) for spec in BUNDLE_MODELS]
    _tar_create(
        bundle_path,
        hub_root=hub,
        model_dirnames=model_dirnames,
        attribution_text=render_attribution(manifest),
        aux_sources=aux_sources,
    )

    # 回填 bundle 整体 sha256 + size
    manifest = manifest.model_copy(update={
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_size": bundle_path.stat().st_size,
    })
    manifest_path.write_text(manifest.model_dump_json(by_alias=True, indent=2) + "\n")

    print(f"\n✅ Bundle 完成")
    print(f"   {bundle_path}  ({bundle_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"   {manifest_path}")
    print(f"\n下一步（在主人有 gh CLI 的机器上）：")
    print(f"   gh release create {bundle_version} \\")
    print(f"     {bundle_path} {manifest_path} \\")
    print(f"     --repo 3Craft/voxkit --notes 'voxkit models bundle {bundle_version}'")
    return int(ExitCode.OK)


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-bundle",
        help="把本机 HF cache 中的 voxkit 必需模型打包成可分发 tar.gz",
    )
    p.add_argument("--hf-cache", default=None,
                   help="HF cache 路径（默认按 HF_HUB_CACHE/HF_HOME/默认路径解析）")
    p.add_argument("--output-dir", default=".",
                   help="bundle + manifest 输出目录（默认当前目录）")
    p.add_argument("--bundle-version", required=True,
                   help='bundle release 标签，例如 "v1" 或 "v0.2.0"')


__all__ = ["run", "add_subparser"]

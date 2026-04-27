"""voxsplit fetch-bundle — 从 GitHub Release 拉模型 bundle 解到本机 HF cache。

数据源优先级：
  1. ``--bundle PATH`` —— 本地已有 tar.gz（开发或离线分发）
  2. ``--from URL`` —— 任意 HTTPS URL
  3. ``--release TAG`` 默认走 GitHub Release（``3Craft/voxsplit``）

GitHub Release 下载策略：
  1. 优先 ``gh release download``（已 auth 的 gh CLI，无需手动配 token）
  2. fallback 到 ``curl + GITHUB_TOKEN``（环境变量）

校验：
  - bundle 整体 sha256 必须与 manifest 一致（下载完即校验，失败立即删）
  - 解压后逐文件 sha256 校验（可 ``--no-verify-files`` 跳过加速）

解压：
  - tar.gz 内部含 ``hub/`` 目录与 ``ATTRIBUTION.md``
  - 解到 HF cache 父目录（``~/.cache/huggingface/``），让 ``hub`` 自然 merge
  - 默认非破坏式：已存在的模型目录跳过；``--force`` 覆盖
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from voxsplit.core.bundle import (
    BundleManifest,
    hf_hub_cache_dir,
    repo_id_to_dirname,
    sha256_file,
)
from voxsplit.core.constants import (
    BUNDLE_FILENAME,
    BUNDLE_GITHUB_REPO,
    BUNDLE_MANIFEST_FILENAME,
    ExitCode,
)


# ── 下载实现 ─────────────────────────────────────────────────────────────
def _download_via_gh(repo: str, tag: str, asset: str, dest: Path) -> bool:
    """用 gh release download 拉 release asset。成功 True；gh 不在或失败 False。"""
    if not shutil.which("gh"):
        return False
    cmd = [
        "gh", "release", "download", tag,
        "--repo", repo,
        "--pattern", asset,
        "--output", str(dest),
        "--clobber",
    ]
    print(f"[fetch] gh release download {tag} {asset}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and dest.is_file():
        return True
    print(f"[fetch] gh 下载失败: {proc.stderr.strip()[:300]}", file=sys.stderr)
    return False


def _download_via_curl(url: str, dest: Path, *, github_token: Optional[str] = None) -> bool:
    """用 curl 拉 URL；如果是 GitHub release asset，自动加 Bearer。

    使用 curl 而非 stdlib 是因为：
    - bundle 30MB+ 需要进度条
    - GitHub release URL 会 302 到 S3，curl 默认 -L 跟随
    """
    if not shutil.which("curl"):
        return False
    cmd = ["curl", "-L", "--fail", "--progress-bar", "-o", str(dest)]
    if github_token and "github.com" in url:
        cmd += ["-H", f"Authorization: Bearer {github_token}",
                "-H", "Accept: application/octet-stream"]
    cmd.append(url)
    print(f"[fetch] curl {url}")
    return subprocess.run(cmd).returncode == 0


def _stdlib_download(url: str, dest: Path) -> bool:
    """fallback：urllib 拉（无进度条），用于 curl 不在场。"""
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        return True
    except Exception as e:
        print(f"[fetch] urllib 下载失败: {e}", file=sys.stderr)
        return False


# ── manifest 拉取（与 bundle 同位置）─────────────────────────────────────
def _fetch_manifest(repo: str, tag: str, dest: Path) -> bool:
    """优先 gh，fallback 到 curl 公共 URL（私有 release 不行，但留作 escape hatch）。"""
    if _download_via_gh(repo, tag, BUNDLE_MANIFEST_FILENAME, dest):
        return True
    # 私有 release 必须 token；公开 release 才能匿名 curl
    print(f"[fetch] gh 不可用且 manifest 在私有 release 中——请确保 gh CLI 已 auth", file=sys.stderr)
    return False


# ── 校验 ─────────────────────────────────────────────────────────────────
def _verify_bundle_sha(bundle: Path, expected: str) -> bool:
    actual = sha256_file(bundle)
    if actual != expected:
        print(f"❌ bundle sha256 不匹配\n   expected: {expected}\n   actual:   {actual}",
              file=sys.stderr)
        return False
    return True


def _verify_files(hub: Path, manifest: BundleManifest, *, sample: bool = False) -> int:
    """逐文件 sha256 校验。``sample=True`` 时只抽查每模型 1 个 blob。返回失败计数。

    bundle 中的 path 字段是相对于 hub 的（含 models--xxx--yyy 目录前缀）。
    """
    failed = 0
    for m in manifest.models:
        real_files = [f for f in m.files if not f.is_symlink]
        targets = real_files[:1] if sample else real_files
        for f in targets:
            fp = hub / f.path
            if not fp.is_file():
                print(f"❌ 缺失: {f.path}", file=sys.stderr)
                failed += 1
                continue
            actual = sha256_file(fp)
            if actual != f.sha256:
                print(f"❌ sha256 不匹配: {f.path}", file=sys.stderr)
                failed += 1
    return failed


# ── 解压（保留 symlinks，幂等 merge）─────────────────────────────────────
def _extract_tarball(bundle: Path, hub: Path, *, force: bool, manifest: BundleManifest) -> None:
    """``tar -xzf`` 解 bundle 直接到 hub/。

    bundle 内顶层是 ``ATTRIBUTION.md`` + 若干 ``models--xxx--yyy/``，正好对应 hub 下的格式。

    非破坏：默认 merge（tar 不会删 destination 中已存在但 tar 内没有的文件）；
    force=True 时先 rmtree 4 个模型目录再解，确保干净。
    """
    hub.mkdir(parents=True, exist_ok=True)

    if force:
        for m in manifest.models:
            d = hub / repo_id_to_dirname(m.repo_id)
            if d.is_dir():
                shutil.rmtree(d)
                print(f"[fetch] --force 删除 {d}")

    cmd = ["tar", "-x", "-z", "-f", str(bundle), "-C", str(hub)]
    subprocess.run(cmd, check=True)


# ── 主流程 ───────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> int:
    hub = Path(args.hf_cache).expanduser().resolve() if args.hf_cache else hf_hub_cache_dir()
    hub.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="voxsplit-fetch-") as tmpd:
        tmp = Path(tmpd)
        bundle_path: Path
        manifest_path: Path

        # ── 决定数据源 ────────────────────────────────────────────────
        if args.bundle:
            bundle_path = Path(args.bundle).expanduser().resolve()
            if not bundle_path.is_file():
                print(f"❌ 本地 bundle 不存在: {bundle_path}", file=sys.stderr)
                return int(ExitCode.GENERIC_FAIL)
            if not args.manifest:
                print("❌ --bundle 必须配合 --manifest 一起用", file=sys.stderr)
                return int(ExitCode.GENERIC_FAIL)
            manifest_path = Path(args.manifest).expanduser().resolve()

        elif args.from_url:
            bundle_path = tmp / BUNDLE_FILENAME
            ok = (_download_via_curl(args.from_url, bundle_path)
                  or _stdlib_download(args.from_url, bundle_path))
            if not ok:
                print(f"❌ 下载失败: {args.from_url}", file=sys.stderr)
                return int(ExitCode.GENERIC_FAIL)
            if not args.manifest:
                print("❌ --from 必须配合 --manifest（验证 sha256）", file=sys.stderr)
                return int(ExitCode.GENERIC_FAIL)
            manifest_path = Path(args.manifest).expanduser().resolve()

        else:
            # 默认走 GitHub Release
            tag = args.release
            repo = args.repo
            bundle_path = tmp / BUNDLE_FILENAME
            manifest_path = tmp / BUNDLE_MANIFEST_FILENAME
            if not _fetch_manifest(repo, tag, manifest_path):
                return int(ExitCode.GENERIC_FAIL)
            if not _download_via_gh(repo, tag, BUNDLE_FILENAME, bundle_path):
                print(f"❌ gh release download 失败；请确认 gh auth login + repo 权限",
                      file=sys.stderr)
                return int(ExitCode.GENERIC_FAIL)

        # ── 解析 manifest ───────────────────────────────────────────
        try:
            manifest = BundleManifest.model_validate_json(manifest_path.read_text())
        except Exception as e:
            print(f"❌ 解析 manifest 失败: {e}", file=sys.stderr)
            return int(ExitCode.GENERIC_FAIL)
        print(f"[fetch] manifest: voxsplit={manifest.voxsplit_version} "
              f"bundle={manifest.bundle_version} models={len(manifest.models)}")

        # ── 校验 bundle 整体 sha256 ────────────────────────────────
        if not _verify_bundle_sha(bundle_path, manifest.bundle_sha256):
            return int(ExitCode.GENERIC_FAIL)
        print(f"[fetch] ✅ bundle sha256 ok ({manifest.bundle_size/1024/1024:.1f} MB)")

        # ── 解压 ────────────────────────────────────────────────────
        _extract_tarball(bundle_path, hub=hub, force=args.force, manifest=manifest)
        print(f"[fetch] ✅ 解压到 {hub}")

        # ── 解压后校验（默认抽样，--verify-all 全量）──────────────
        if not args.no_verify:
            failed = _verify_files(hub, manifest, sample=not args.verify_all)
            if failed:
                print(f"❌ {failed} 个文件 sha256 校验失败", file=sys.stderr)
                return int(ExitCode.GENERIC_FAIL)
            print(f"[fetch] ✅ 文件校验通过"
                  + ("（全量）" if args.verify_all else "（抽样）"))

    print(f"\n✅ 模型已就绪。运行 'voxsplit doctor' 验证；可直接 'voxsplit diarize ...'")
    return int(ExitCode.OK)


def add_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "fetch-bundle",
        help="从 GitHub Release 拉模型 bundle 解到本机 HF cache",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--release", default="latest",
                     help="GitHub Release tag（默认 latest）")
    src.add_argument("--from", dest="from_url",
                     help="任意 HTTPS URL（需配 --manifest）")
    src.add_argument("--bundle",
                     help="本地 .tar.gz 路径（需配 --manifest）")

    p.add_argument("--manifest",
                   help="本地 manifest.json 路径（与 --bundle / --from 配对）")
    p.add_argument("--repo", default=BUNDLE_GITHUB_REPO,
                   help=f"GitHub repo（默认 {BUNDLE_GITHUB_REPO}）")
    p.add_argument("--hf-cache", default=None,
                   help="HF cache 路径（默认按 HF_HUB_CACHE/HF_HOME 解析）")
    p.add_argument("--force", action="store_true",
                   help="覆盖已存在的模型目录")
    p.add_argument("--no-verify", action="store_true",
                   help="跳过解压后文件校验")
    p.add_argument("--verify-all", action="store_true",
                   help="全量文件 sha256 校验（默认每模型抽样 1 个文件）")


__all__ = ["run", "add_subparser"]

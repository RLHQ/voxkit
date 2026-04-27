"""bundle 模块端到端测试：fake HF cache → build → fetch round-trip。

策略：
- 用 ``tmp_path`` 构造一个模拟的 HF cache，含一个伪 repo 目录（含 blobs +
  snapshots/<commit>/file 的 symlink + refs/main）
- monkey-patch ``BUNDLE_MODELS`` 只指向这个伪 repo（避免依赖真实的 4 个模型）
- 调用 ``build_bundle.run`` 生成 .tar.gz + manifest.json
- 调用 ``fetch_bundle.run --bundle ... --manifest ...`` 解到另一个空 cache
- 校验：refs / snapshots symlink / blob 内容 / sha256 全对
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from voxsplit.commands import build_bundle as B
from voxsplit.commands import fetch_bundle as F
from voxsplit.core import bundle as core_bundle


# ── fixture：fake HF cache ─────────────────────────────────────────────
def _make_fake_cache(root: Path, repo_id: str, *, commit: str, files: dict[str, bytes]) -> Path:
    """在 root 下创建 hub/<repo-dirname>/ 目录，含 blobs / snapshots / refs。

    files: {filename: content_bytes}
    """
    hub = root / "hub"
    repo_dir = hub / core_bundle.repo_id_to_dirname(repo_id)
    blobs = repo_dir / "blobs"
    snap = repo_dir / "snapshots" / commit
    refs = repo_dir / "refs"
    for d in (blobs, snap, refs):
        d.mkdir(parents=True)

    (refs / "main").write_text(commit)

    for name, data in files.items():
        sha = hashlib.sha256(data).hexdigest()
        blob = blobs / sha
        blob.write_bytes(data)
        # snapshot symlink 用 HF 标准的相对路径 "../../blobs/<sha>"
        link = snap / name
        rel_target = Path("..") / ".." / "blobs" / sha
        os.symlink(rel_target, link)
    return hub


# ── monkey-patch BUNDLE_MODELS / 默认 cache ────────────────────────────
@pytest.fixture
def patched_bundle_models(monkeypatch):
    """让 build / doctor 只关心一个伪 repo，避免触碰真实 4 个模型。"""
    fake = [{"repo_id": "fake-org/fake-model", "license": "mit"}]
    monkeypatch.setattr("voxsplit.commands.build_bundle.BUNDLE_MODELS", fake)
    return fake


# ── 测试 ───────────────────────────────────────────────────────────────
def test_hf_hub_cache_dir_env_priority(tmp_path, monkeypatch):
    """HF_HUB_CACHE > HF_HOME/hub > 默认。"""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    # 默认 ~/.cache/huggingface/hub
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert core_bundle.hf_hub_cache_dir() == tmp_path / ".cache" / "huggingface" / "hub"

    # HF_HOME → HF_HOME/hub
    monkeypatch.setenv("HF_HOME", str(tmp_path / "alt-hf"))
    assert core_bundle.hf_hub_cache_dir() == tmp_path / "alt-hf" / "hub"

    # HF_HUB_CACHE 最高优先
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "explicit-cache"))
    assert core_bundle.hf_hub_cache_dir() == tmp_path / "explicit-cache"


def test_repo_id_to_dirname():
    assert core_bundle.repo_id_to_dirname("pyannote/segmentation-3.0") == \
        "models--pyannote--segmentation-3.0"


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert core_bundle.sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_build_then_fetch_round_trip(tmp_path, patched_bundle_models):
    """完整 round-trip：build 出 bundle → fetch 到另一个 cache → 内容一致。"""
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    out_dir = tmp_path / "release-staging"
    out_dir.mkdir()

    # 构造源 cache：一个伪 repo，3 个文件
    files = {
        "config.yaml": b"key: value\nnum: 42\n",
        "pytorch_model.bin": b"\x00\x01\x02\x03" * 1024,  # 4 KB 假权重
        "README.md": b"# fake model\n",
    }
    src_hub = _make_fake_cache(src_root, "fake-org/fake-model",
                               commit="deadbeef" * 5, files=files)

    # ── build ───────────────────────────────────────────────────────
    rc = B.run(argparse.Namespace(
        hf_cache=str(src_hub),
        output_dir=str(out_dir),
        bundle_version="v-test",
    ))
    assert rc == 0
    bundle_p = out_dir / "voxsplit-models.tar.gz"
    manifest_p = out_dir / "voxsplit-models.manifest.json"
    assert bundle_p.is_file()
    assert manifest_p.is_file()

    manifest = core_bundle.BundleManifest.model_validate_json(manifest_p.read_text())
    assert manifest.bundle_version == "v-test"
    assert len(manifest.models) == 1
    assert manifest.models[0].repo_id == "fake-org/fake-model"
    # 3 个 snapshot symlinks + 3 个 blobs + 1 个 refs/main = 7 个 files
    assert len(manifest.models[0].files) == 7
    # bundle 整体 sha256 已回填
    assert manifest.bundle_sha256 == core_bundle.sha256_file(bundle_p)

    # ── fetch（用 --bundle / --manifest 模式，跳过网络）────────
    dst_hub = dst_root / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(bundle_p),
        manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub),
        force=False, no_verify=False, verify_all=True,
    ))
    assert rc == 0
    # 副作用断言：tar 不应当在 dst_hub 同级创建 hub/（旧 bug）
    assert not (dst_hub / "hub").exists(), \
        "bundle 顶层不应再有 hub/ 包装层"

    # ── 校验解压结果与源一致 ────────────────────────────────
    repo_dirname = core_bundle.repo_id_to_dirname("fake-org/fake-model")
    dst_repo = dst_hub / repo_dirname
    assert (dst_repo / "refs" / "main").read_text() == "deadbeef" * 5

    snap = dst_repo / "snapshots" / ("deadbeef" * 5)
    for name, data in files.items():
        link = snap / name
        # symlink 必须保留（不能被解压成普通文件副本）
        assert link.is_symlink(), f"{name} 解压后失去 symlink"
        # symlink 指向的 blob 内容应一致
        assert link.read_bytes() == data


def test_build_fails_on_missing_refs(tmp_path, patched_bundle_models):
    """refs/main 缺失或为空 → build 应报错并 exit 1。"""
    src_root = tmp_path / "src"
    src_hub = src_root / "hub"
    repo_dir = src_hub / core_bundle.repo_id_to_dirname("fake-org/fake-model")
    (repo_dir / "snapshots" / "abc").mkdir(parents=True)
    (repo_dir / "blobs").mkdir()
    # 故意不写 refs/main

    out = tmp_path / "out"
    out.mkdir()
    rc = B.run(argparse.Namespace(
        hf_cache=str(src_hub),
        output_dir=str(out),
        bundle_version="v-test",
    ))
    assert rc == 1


def test_fetch_detects_corrupt_bundle(tmp_path, patched_bundle_models):
    """bundle sha256 与 manifest 不一致 → fetch 应失败。"""
    src_root = tmp_path / "src"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    src_hub = _make_fake_cache(src_root, "fake-org/fake-model",
                               commit="abcdef" * 6, files={"a.txt": b"hi"})
    assert B.run(argparse.Namespace(
        hf_cache=str(src_hub),
        output_dir=str(out_dir),
        bundle_version="v-test",
    )) == 0

    bundle_p = out_dir / "voxsplit-models.tar.gz"
    manifest_p = out_dir / "voxsplit-models.manifest.json"

    # 故意篡改 bundle（追加 1 字节）
    with bundle_p.open("ab") as f:
        f.write(b"X")

    dst_hub = tmp_path / "dst" / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(bundle_p),
        manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub),
        force=False, no_verify=False, verify_all=False,
    ))
    assert rc == 1

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

from voxkit.commands import build_bundle as B
from voxkit.commands import fetch_bundle as F
from voxkit.core import bundle as core_bundle


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
    """让 build / doctor 只关心一个伪 repo，避免触碰真实 4 个模型。

    同时把 ``BUNDLE_AUX_FILES`` 默认 patch 为空——保证现有测试在跑 build/fetch
    时不会触碰开发者机器上真实的 /opt/homebrew/.../silero.bin，
    更不会把 aux 文件落到真实 ``~/.cache/voxkit/aux/``。
    需要测 aux 行为的新测试自己再 patch 该常量。
    """
    fake = [{"repo_id": "fake-org/fake-model", "license": "mit"}]
    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_MODELS", fake)
    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_AUX_FILES", [])
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


def test_models_offline_ready_all_present(tmp_path, monkeypatch):
    """4 个 BUNDLE_MODELS 都有 refs/main + 非空 snapshot → True。"""
    fake_specs = [
        {"repo_id": "fake/m1", "license": "mit"},
        {"repo_id": "fake/m2", "license": "mit"},
    ]
    monkeypatch.setattr("voxkit.core.bundle.BUNDLE_MODELS", fake_specs, raising=False)
    # bundle.models_offline_ready 用 lazy import，要 patch 真正的 source
    import voxkit.core.constants as C
    monkeypatch.setattr(C, "BUNDLE_MODELS", fake_specs)

    hub = tmp_path / "hub"
    for spec in fake_specs:
        _make_fake_cache(tmp_path / spec["repo_id"].replace("/", "_"),
                         spec["repo_id"], commit="c" * 12,
                         files={"x.bin": b"hi"})
    # 把所有伪 repo 的 hub/ 拷贝合并到一个总 hub/
    hub.mkdir()
    for spec in fake_specs:
        sub = (tmp_path / spec["repo_id"].replace("/", "_") / "hub")
        for child in sub.iterdir():
            shutil.copytree(child, hub / child.name)

    assert core_bundle.models_offline_ready(hub) is True


def test_models_offline_ready_missing_one(tmp_path, monkeypatch):
    """任一模型缺 → False。"""
    fake_specs = [
        {"repo_id": "fake/present", "license": "mit"},
        {"repo_id": "fake/missing", "license": "mit"},
    ]
    import voxkit.core.constants as C
    monkeypatch.setattr(C, "BUNDLE_MODELS", fake_specs)

    hub = tmp_path / "hub"
    src_root = tmp_path / "src"
    _make_fake_cache(src_root, "fake/present", commit="a" * 12,
                     files={"x.bin": b"hi"})
    hub.mkdir()
    for child in (src_root / "hub").iterdir():
        shutil.copytree(child, hub / child.name)
    # fake/missing 故意不创建

    assert core_bundle.models_offline_ready(hub) is False


def test_models_offline_ready_empty_refs(tmp_path, monkeypatch):
    """refs/main 存在但内容为空 → False（防御 build 中断的半成品 cache）。"""
    fake_specs = [{"repo_id": "fake/empty-refs", "license": "mit"}]
    import voxkit.core.constants as C
    monkeypatch.setattr(C, "BUNDLE_MODELS", fake_specs)

    hub = tmp_path / "hub"
    repo_dir = hub / core_bundle.repo_id_to_dirname("fake/empty-refs")
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text("")  # 空内容
    (repo_dir / "snapshots" / "anything").mkdir(parents=True)
    (repo_dir / "snapshots" / "anything" / "x.txt").write_text("y")

    assert core_bundle.models_offline_ready(hub) is False


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
    bundle_p = out_dir / "voxkit-models.tar.gz"
    manifest_p = out_dir / "voxkit-models.manifest.json"
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

    bundle_p = out_dir / "voxkit-models.tar.gz"
    manifest_p = out_dir / "voxkit-models.manifest.json"

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


# ── aux files 测试（Round 2 Agent B）────────────────────────────────────
def _make_aux_spec(tmp_path: Path, *, src_filename: str, target_filename: str,
                   content: bytes) -> tuple[Path, dict]:
    """构造一个 aux spec dict + 写入源文件。返回 (target_path, spec)。"""
    src = tmp_path / "fake-brew" / src_filename
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(content)
    target = tmp_path / "fake-target" / target_filename
    spec = {
        "name": "fake-aux",
        "filename": src_filename,  # bundle tar 内 aux/<filename>
        "license": "mit",
        "source_candidates": [str(src)],
        "target": str(target),
    }
    return target, spec


def _build_simple_bundle(
    tmp_path: Path, *, aux_specs: list[dict] | None = None
) -> tuple[Path, Path]:
    """build 出一个最小 bundle（1 个伪 model，0 或多个 aux）。返回 (bundle, manifest)。

    通过 monkey-patch 临时替换 build_bundle 模块级 BUNDLE_AUX_FILES。需要在
    fixture 已经把 BUNDLE_MODELS 替换为伪 repo 之后调用。
    """
    src_root = tmp_path / "src"
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    src_hub = _make_fake_cache(src_root, "fake-org/fake-model",
                               commit="abc123" * 6,
                               files={"config.yaml": b"k: v\n"})

    # mp 的 setattr 必须用 monkeypatch，但这函数没有 fixture——交给调用方用 monkeypatch.setattr
    rc = B.run(argparse.Namespace(
        hf_cache=str(src_hub),
        output_dir=str(out_dir),
        bundle_version="v-aux",
    ))
    assert rc == 0
    return out_dir / "voxkit-models.tar.gz", out_dir / "voxkit-models.manifest.json"


def test_build_with_aux_file_present(tmp_path, monkeypatch, patched_bundle_models):
    """aux 文件源存在 → manifest 含 aux_files 且 tar 包含 aux/<filename>。"""
    import tarfile

    target, spec = _make_aux_spec(
        tmp_path,
        src_filename="ggml-silero-fake.bin",
        target_filename="ggml-silero-fake.bin",
        content=b"\x10" * 2048,
    )
    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_AUX_FILES", [spec])

    bundle_p, manifest_p = _build_simple_bundle(tmp_path)
    manifest = core_bundle.BundleManifest.model_validate_json(manifest_p.read_text())

    assert len(manifest.aux_files) == 1
    aux = manifest.aux_files[0]
    assert aux.filename == "ggml-silero-fake.bin"
    assert aux.size_bytes == 2048
    assert aux.sha256 == hashlib.sha256(b"\x10" * 2048).hexdigest()
    # target_path 应已展开（无 ~），与 spec 中 tmp 路径一致
    assert aux.target_path == str(target)

    # tar 内必须有 aux/<filename>
    with tarfile.open(bundle_p, "r:gz") as tf:
        names = tf.getnames()
    assert "aux/ggml-silero-fake.bin" in names


def test_build_with_aux_file_missing(tmp_path, monkeypatch, patched_bundle_models, capsys):
    """所有 source_candidates 不存在 → build 不报错，aux_files 为空，stderr 有警告。"""
    spec = {
        "name": "missing-aux",
        "filename": "nonexistent.bin",
        "license": "mit",
        "source_candidates": [
            str(tmp_path / "definitely-not-here-1.bin"),
            str(tmp_path / "definitely-not-here-2.bin"),
        ],
        "target": str(tmp_path / "target" / "nonexistent.bin"),
    }
    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_AUX_FILES", [spec])

    bundle_p, manifest_p = _build_simple_bundle(tmp_path)
    manifest = core_bundle.BundleManifest.model_validate_json(manifest_p.read_text())

    assert manifest.aux_files == []  # 空列表（兼容 pydantic 默认）
    captured = capsys.readouterr()
    assert "not found" in captured.err
    assert "nonexistent.bin" in captured.err


def test_fetch_installs_aux_file_to_target(tmp_path, monkeypatch, patched_bundle_models):
    """build → fetch round-trip：aux 文件落到 target_path 且 sha256 一致。"""
    content = b"silero-fake-payload" * 64  # 1216 bytes
    target, spec = _make_aux_spec(
        tmp_path,
        src_filename="ggml-silero-fake.bin",
        target_filename="ggml-silero-fake.bin",
        content=content,
    )
    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_AUX_FILES", [spec])

    bundle_p, manifest_p = _build_simple_bundle(tmp_path)

    dst_hub = tmp_path / "dst" / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(bundle_p), manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub),
        force=False, no_verify=False, verify_all=True,
    ))
    assert rc == 0
    # 校验 target 落地
    assert target.is_file(), f"aux 应被搬到 {target}"
    assert target.read_bytes() == content
    # staging hub/aux/ 必须被清理（不留垃圾）
    assert not (dst_hub / "aux").exists(), "hub/aux/ 必须被清理"


def test_fetch_backward_compat_no_aux(tmp_path, patched_bundle_models):
    """没有 aux_files 的 bundle（fixture 默认空）→ build/fetch 都正常。"""
    src_root = tmp_path / "src"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src_hub = _make_fake_cache(src_root, "fake-org/fake-model",
                               commit="0" * 30, files={"a.txt": b"hi\n"})

    rc = B.run(argparse.Namespace(
        hf_cache=str(src_hub),
        output_dir=str(out_dir),
        bundle_version="v-noaux",
    ))
    assert rc == 0
    bundle_p = out_dir / "voxkit-models.tar.gz"
    manifest_p = out_dir / "voxkit-models.manifest.json"

    manifest = core_bundle.BundleManifest.model_validate_json(manifest_p.read_text())
    assert manifest.aux_files == []  # 空列表

    # JSON 中即便有 auxFiles 字段为空数组也是合法的（字段允许默认值）
    raw = json.loads(manifest_p.read_text())
    assert raw.get("auxFiles", []) == []

    # fetch 应当不抛错且不创建 hub/aux/
    dst_hub = tmp_path / "dst" / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(bundle_p), manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub),
        force=False, no_verify=False, verify_all=True,
    ))
    assert rc == 0
    assert not (dst_hub / "aux").exists()


def test_fetch_old_manifest_without_aux_field(tmp_path, patched_bundle_models):
    """显式删除 manifest 中的 auxFiles 字段（模拟 v0.2.0 旧 bundle）→ fetch 仍然成功。"""
    src_root = tmp_path / "src"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    src_hub = _make_fake_cache(src_root, "fake-org/fake-model",
                               commit="1" * 30, files={"a.txt": b"hi\n"})

    rc = B.run(argparse.Namespace(
        hf_cache=str(src_hub),
        output_dir=str(out_dir),
        bundle_version="v-legacy",
    ))
    assert rc == 0
    manifest_p = out_dir / "voxkit-models.manifest.json"

    # 模拟旧版 manifest：删掉 auxFiles 键
    raw = json.loads(manifest_p.read_text())
    raw.pop("auxFiles", None)
    manifest_p.write_text(json.dumps(raw))

    # 仍然要能解析 + fetch
    parsed = core_bundle.BundleManifest.model_validate_json(manifest_p.read_text())
    assert parsed.aux_files == []

    dst_hub = tmp_path / "dst" / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(out_dir / "voxkit-models.tar.gz"),
        manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub),
        force=False, no_verify=False, verify_all=True,
    ))
    assert rc == 0


def test_build_aux_target_path_keeps_tilde_for_portability(
    tmp_path, monkeypatch, patched_bundle_models
):
    """spec 用 ``~/...`` → manifest 必须保留 ``~/...``，不能展开成 builder 的绝对路径。

    回归测试：之前 build 端调用 expanduser() 把 ``~/.cache/voxkit/aux/x.bin`` 烤成
    ``/Users/<builder>/.cache/voxkit/aux/x.bin``，导致其他用户拉到 release 后
    试图写到不存在的路径。修复：原样存 ``~/...``，让 fetch 端按当前 ``Path.home()``
    展开。
    """
    src = tmp_path / "fake-brew" / "ggml-silero-fake.bin"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"\x10" * 256)

    spec = {
        "name": "silero-vad",
        "filename": "ggml-silero-fake.bin",
        "license": "mit",
        "source_candidates": [str(src)],
        # 关键：spec 用 ~ 形式，模拟真实 BUNDLE_AUX_FILES
        "target": "~/.cache/voxkit/aux/ggml-silero-fake.bin",
    }
    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_AUX_FILES", [spec])

    _bundle_p, manifest_p = _build_simple_bundle(tmp_path)
    manifest = core_bundle.BundleManifest.model_validate_json(manifest_p.read_text())

    assert len(manifest.aux_files) == 1
    aux = manifest.aux_files[0]
    # 必须保留 ~ 前缀；绝对不能含 builder 的 home 路径
    assert aux.target_path == "~/.cache/voxkit/aux/ggml-silero-fake.bin", (
        f"target_path 不应被 build 端 expanduser；得到: {aux.target_path}"
    )
    assert "/Users/" not in aux.target_path
    assert "/home/" not in aux.target_path


def test_fetch_aux_with_tilde_target_path_uses_runtime_home(
    tmp_path, monkeypatch, patched_bundle_models
):
    """manifest 中 ``~/...`` → fetch 端按运行时 ``Path.home()`` 展开，不用 builder 的 home。

    模拟：build 在 user-A 机器打的 bundle，fetch 在 user-B 机器跑——target 必须落
    到 user-B 的 home，不是被烤进 manifest 的某个固定路径。
    """
    src = tmp_path / "fake-brew" / "aux.bin"
    src.parent.mkdir(parents=True, exist_ok=True)
    payload = b"portable-payload" * 16
    src.write_bytes(payload)

    spec = {
        "name": "fake-aux",
        "filename": "aux.bin",
        "license": "mit",
        "source_candidates": [str(src)],
        "target": "~/.cache/voxkit/aux/aux.bin",
    }
    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_AUX_FILES", [spec])

    bundle_p, manifest_p = _build_simple_bundle(tmp_path)

    # ── fetch 时把 $HOME 重定向到 tmp_path（模拟另一个用户）──
    # 注意：Path.expanduser() 走的是 os.path.expanduser → $HOME env，
    # 不是 Path.home()。monkeypatch Path.home 没用，必须 setenv HOME。
    fake_home = tmp_path / "fake-home-of-user-B"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    dst_hub = tmp_path / "dst" / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(bundle_p), manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub),
        force=False, no_verify=False, verify_all=True,
    ))
    assert rc == 0
    expected = fake_home / ".cache" / "voxkit" / "aux" / "aux.bin"
    assert expected.is_file(), f"aux 应被解析到运行时 home: {expected}"
    assert expected.read_bytes() == payload


def test_fetch_aux_skip_without_force(tmp_path, monkeypatch, patched_bundle_models, capsys):
    """target 已存在 + 无 --force → 跳过；带 --force → 覆盖。"""
    content_v1 = b"first-version"
    content_v2 = b"second-version-different-bytes"
    target, spec = _make_aux_spec(
        tmp_path,
        src_filename="aux.bin",
        target_filename="aux.bin",
        content=content_v2,  # bundle 中是 v2
    )
    # 先用 v1 占位 target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content_v1)
    pre_mtime = target.stat().st_mtime

    monkeypatch.setattr("voxkit.commands.build_bundle.BUNDLE_AUX_FILES", [spec])
    bundle_p, manifest_p = _build_simple_bundle(tmp_path)

    # 第一次 fetch：no force → 跳过，target 保持 v1
    dst_hub = tmp_path / "dst1" / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(bundle_p), manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub),
        force=False, no_verify=False, verify_all=True,
    ))
    assert rc == 0
    assert target.read_bytes() == content_v1, "no force 不应覆盖现存 target"
    captured = capsys.readouterr()
    assert "skipped" in captured.out

    # 第二次 fetch：force=True → 覆盖为 v2
    dst_hub2 = tmp_path / "dst2" / "hub"
    rc = F.run(argparse.Namespace(
        bundle=str(bundle_p), manifest=str(manifest_p),
        from_url=None, release=None, repo="any/any",
        hf_cache=str(dst_hub2),
        force=True, no_verify=False, verify_all=True,
    ))
    assert rc == 0
    assert target.read_bytes() == content_v2, "--force 应覆盖现存 target"

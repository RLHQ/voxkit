"""模型 bundle 共享模块：schema、HF cache 路径解析、CC-BY-4.0 attribution 模板。

bundle 的设计原则：
- 直接预填充 HF cache 目录（pyannote/huggingface_hub 命中即跳过网络）
- bundle 内部保留 HF 标准结构（blobs / snapshots / refs），symlinks 保留
- manifest.json 含每文件 sha256 + 每模型 commit hash + license / source URL
- ATTRIBUTION.md 是 CC-BY-4.0 强制要求（speaker-diarization-community-1 与
  wespeaker 模型）。bundle 同名同 license 重发可，但必须保留版权声明。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── HF cache 路径探测（复刻 huggingface_hub.constants 的 fallback 链）───────
def hf_hub_cache_dir() -> Path:
    """返回 HF Hub 模型 cache 目录。优先级：

      1. ``HF_HUB_CACHE`` env
      2. ``HF_HOME/hub``
      3. ``~/.cache/huggingface/hub``

    huggingface_hub 1.x 完全按此顺序解析；本函数复刻以避免主进程依赖 hf_hub。
    """
    if v := os.environ.get("HF_HUB_CACHE"):
        return Path(v).expanduser()
    if v := os.environ.get("HF_HOME"):
        return Path(v).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_id_to_dirname(repo_id: str) -> str:
    """``pyannote/speaker-diarization-3.1`` → ``models--pyannote--speaker-diarization-3.1``。"""
    return "models--" + repo_id.replace("/", "--")


# ── manifest schema ────────────────────────────────────────────────────────
class FileEntry(BaseModel):
    """bundle 中单个文件的元数据。"""

    model_config = ConfigDict(populate_by_name=True)

    path: str = Field(..., description="bundle 内相对路径，例如 'models--pyannote--xxx/blobs/<sha>'")
    size: int
    sha256: str
    is_symlink: bool = Field(False, alias="isSymlink")
    symlink_target: Optional[str] = Field(
        None,
        alias="symlinkTarget",
        description="symlink 时记录目标（相对路径），fetch 解压后用于校验",
    )


class ModelEntry(BaseModel):
    """bundle 中一个模型仓库的元数据。"""

    model_config = ConfigDict(populate_by_name=True)

    repo_id: str = Field(..., alias="repoId", description='例如 "pyannote/speaker-diarization-3.1"')
    commit: str = Field(..., description="HF refs/main 中的 commit hash")
    license: str = Field(..., description='例如 "mit" / "cc-by-4.0"')
    source_url: str = Field(..., alias="sourceUrl")
    total_size: int = Field(..., alias="totalSize")
    files: List[FileEntry]


class AuxFileEntry(BaseModel):
    """bundle 中辅助文件（非 HF repo）的元数据。

    与 ``ModelEntry`` 不同，aux 文件没有 commit / snapshot / blobs 结构，就是单一文件，
    在 tar 内放在 ``aux/<filename>``，fetch 时按 ``target_path`` 落到用户机器。
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., description="bundle 内部稳定标识，例如 silero-vad")
    filename: str = Field(..., description="文件名，对应 tar 内 aux/<filename>")
    license: str = Field(..., description='例如 "mit"')
    sha256: str
    size_bytes: int = Field(..., alias="sizeBytes")
    target_path: str = Field(
        ...,
        alias="targetPath",
        description="fetch-bundle 解压目标路径（已展开 ~，绝对路径）",
    )


class BundleManifest(BaseModel):
    """voxkit 模型 bundle 顶层 manifest，作为单独 release asset 与 .tar.gz 一起发布。"""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field("1", alias="schemaVersion")
    voxkit_version: str = Field(..., alias="voxkitVersion")
    bundle_version: str = Field(..., alias="bundleVersion", description="bundle release tag，如 v1")
    created_at: str = Field(..., alias="createdAt", description="ISO-8601 UTC")
    bundle_filename: str = Field(..., alias="bundleFilename")
    bundle_sha256: str = Field(..., alias="bundleSha256", description="整个 .tar.gz 的 sha256")
    bundle_size: int = Field(..., alias="bundleSize")
    models: List[ModelEntry]
    # aux_files 是新增字段；旧 bundle manifest 中没有此 key，反序列化时取默认空列表，保持向后兼容
    aux_files: List[AuxFileEntry] = Field(default_factory=list, alias="auxFiles")


# ── 帮助函数 ───────────────────────────────────────────────────────────────
def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 sha256（支持大文件不爆内存）。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ── CC-BY-4.0 attribution 模板 ────────────────────────────────────────────
ATTRIBUTION_TEMPLATE = """# voxkit Models Bundle — Attribution

This bundle redistributes pre-trained model weights from Hugging Face. All
weights remain the property of their respective authors and are licensed
under the terms below.

By unpacking this bundle into your local Hugging Face cache, you agree to
the upstream model licenses.

## Models Included

{model_blocks}

## Notes

- The `cc-by-4.0` models above require **attribution** when redistributed.
  This file fulfils that requirement; please keep it next to any further
  copies of the bundle.
- All models are unmodified copies of the upstream commits referenced
  above. Their behaviour, safety properties, and limitations are entirely
  inherited from upstream.
- This bundle was produced by `voxkit build-bundle` (voxkit
  {voxkit_version}, bundle {bundle_version}).
"""

_MODEL_BLOCK = """### `{repo_id}`

- **Source**: {source_url}
- **Commit**: `{commit}`
- **License**: `{license}`
- **Size**: {size_mb:.2f} MB
"""

_AUX_BLOCK = """### `{name}` ({filename})

- **License**: `{license}`
- **Size**: {size_mb:.2f} MB
- **Installed to**: `{target_path}`
"""


def render_attribution(manifest: BundleManifest) -> str:
    """根据 manifest 生成 ATTRIBUTION.md 全文。"""
    blocks = []
    for m in manifest.models:
        blocks.append(_MODEL_BLOCK.format(
            repo_id=m.repo_id,
            source_url=m.source_url,
            commit=m.commit,
            license=m.license,
            size_mb=m.total_size / (1024 * 1024),
        ))
    # aux 文件单独成段，license attribution 同样必要（即便都是 MIT）
    for a in manifest.aux_files:
        blocks.append(_AUX_BLOCK.format(
            name=a.name,
            filename=a.filename,
            license=a.license,
            size_mb=a.size_bytes / (1024 * 1024),
            target_path=a.target_path,
        ))
    return ATTRIBUTION_TEMPLATE.format(
        model_blocks="\n".join(blocks).rstrip(),
        voxkit_version=manifest.voxkit_version,
        bundle_version=manifest.bundle_version,
    )


__all__ = [
    "hf_hub_cache_dir",
    "repo_id_to_dirname",
    "sha256_file",
    "FileEntry",
    "ModelEntry",
    "AuxFileEntry",
    "BundleManifest",
    "render_attribution",
    "ATTRIBUTION_TEMPLATE",
]

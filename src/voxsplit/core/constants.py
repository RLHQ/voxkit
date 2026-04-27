"""voxsplit 跨模块共享常量：模型/路径/exit code。

所有原本散落在 cli.py / commands/ / lazy_install.py / doctor.py 的字符串与
路径在此唯一化，新增模型 / 改 venv 位置只需改这一个文件。
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

# ── 模型别名 ─────────────────────────────────────────────────
# 短名是 CLI 暴露的稳定名；全名是 HF repo id
MODEL_ALIASES: dict[str, str] = {
    "sd-3.1":      "pyannote/speaker-diarization-3.1",
    "community-1": "pyannote/speaker-diarization-community-1",
}
MODEL_CHOICES: list[str] = list(MODEL_ALIASES.keys())
DEFAULT_MODEL = "sd-3.1"

# ── HF gated 自检：实际权重文件 URL（HEAD metadata 200 不等于 accept）─
GATED_WEIGHTS: list[tuple[str, str]] = [
    (MODEL_ALIASES["sd-3.1"],
     f"https://huggingface.co/{MODEL_ALIASES['sd-3.1']}/resolve/main/config.yaml"),
    ("pyannote/segmentation-3.0",
     "https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin"),
    (MODEL_ALIASES["community-1"],
     f"https://huggingface.co/{MODEL_ALIASES['community-1']}/resolve/main/config.yaml"),
    ("pyannote/wespeaker-voxceleb-resnet34-LM",
     "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin"),
]

# ── HF token 路径（按优先级）──────────────────────────────
HF_TOKEN_PATHS: list[Path] = [
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
]

# ── voxsplit 用户级 venv ──────────────────────────────────
VENV_DIR: Path = Path.home() / ".local" / "share" / "voxsplit" / "venv"
VENV_PYTHON: Path = VENV_DIR / "bin" / "python"
INSTALLED_MARKER: Path = Path.home() / ".cache" / "voxsplit" / ".installed"

# pyannote.audio 版本（与 pyproject.toml [worker] 一致）
PYANNOTE_VERSION_SPEC = "pyannote.audio>=4.0.4,<5"

# Worker stdout JSON sentinel（避开 torch/pyannote 偶发 print 干扰）
WORKER_JSON_SENTINEL = "__VOXSPLIT_JSON__"


# ── Exit codes（只追加，不重排）────────────────────────────
class ExitCode(IntEnum):
    OK = 0
    GENERIC_FAIL = 1
    HF_AUTH = 2          # token 缺失 / gated 未 accept
    ENV_PROBLEM = 3      # ffmpeg / libavutil
    WORKER_FAILED = 4    # pyannote 推理出错 / OOM / fallback
    INTERRUPTED = 5      # SIGINT


__all__ = [
    "MODEL_ALIASES", "MODEL_CHOICES", "DEFAULT_MODEL",
    "GATED_WEIGHTS",
    "HF_TOKEN_PATHS",
    "VENV_DIR", "VENV_PYTHON", "INSTALLED_MARKER",
    "PYANNOTE_VERSION_SPEC",
    "WORKER_JSON_SENTINEL",
    "ExitCode",
]

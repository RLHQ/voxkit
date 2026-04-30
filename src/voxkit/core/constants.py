"""voxkit 跨模块共享常量：模型/路径/exit code。

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

# bundle 必备的所有模型 repo（含 license，离线 attribution 用）
BUNDLE_MODELS: list[dict] = [
    {"repo_id": MODEL_ALIASES["sd-3.1"],            "license": "mit"},
    {"repo_id": "pyannote/segmentation-3.0",        "license": "mit"},
    {"repo_id": MODEL_ALIASES["community-1"],       "license": "cc-by-4.0"},
    {"repo_id": "pyannote/wespeaker-voxceleb-resnet34-LM", "license": "cc-by-4.0"},
]

# ── whisper.cpp aux 文件路径（brew 自带 / voxkit 自有 cache）─────────
# 集中所有"silero VAD 文件名 + brew 安装位置"硬编码，避免散在 constants /
# whisper_exec / build_bundle / 测试里六处。
SILERO_VAD_FILENAME = "ggml-silero-v5.1.2.bin"
WHISPER_CPP_BREW_SHARE_APPLE = "/opt/homebrew/share/whisper-cpp"  # Apple Silicon brew
WHISPER_CPP_BREW_SHARE_INTEL = "/usr/local/share/whisper-cpp"     # Intel mac / Linux brew

# voxkit 自有 cache 目录（与 HF cache 解耦；fetch-bundle 用 ~ 形式存进 manifest，
# 保证跨用户/跨机器可移植）
VOXKIT_AUX_DIR_TILDE = "~/.cache/voxkit/aux"


# 与 HF 模型一同打包的辅助文件（不是 HF repo，本身可以独立分发）。
# 每条目语义：
#   name              — bundle 内部稳定标识（不是文件名）
#   filename          — bundle tar 内 aux/<filename> 与 manifest 中的文件名
#   license           — 用于 ATTRIBUTION 渲染
#   source_candidates — build-bundle 在开发者机器上按顺序探测，第一个存在的入选
#   target            — fetch-bundle 解到用户机器的目标路径（保留 ~ 形式）
#
# 当前唯一辅助文件：silero VAD（whisper.cpp 用，约 ~2MB），用于反幻觉。
# whisper.cpp 模型本身因许可 + 体积留在 brew/HF；doctor 给用户安装提示。
BUNDLE_AUX_FILES: list[dict] = [
    {
        "name": "silero-vad",
        "filename": SILERO_VAD_FILENAME,
        "license": "mit",
        "source_candidates": [
            f"{WHISPER_CPP_BREW_SHARE_APPLE}/{SILERO_VAD_FILENAME}",
            f"{WHISPER_CPP_BREW_SHARE_INTEL}/{SILERO_VAD_FILENAME}",
        ],
        "target": f"{VOXKIT_AUX_DIR_TILDE}/{SILERO_VAD_FILENAME}",
    },
]


# ── HF Hub 环境变量 ──────────────────────────────────────────────────
# huggingface_hub 官方约定：设为 "1" → 完全跳过 HEAD 请求，纯走本地 cache。
# patched_env() 在模型 cache 齐全时自动注入此变量。
HF_HUB_OFFLINE_ENV = "HF_HUB_OFFLINE"

# bundle 默认 GitHub Release（私有 repo，下载需 PAT 或 gh CLI auth）
BUNDLE_GITHUB_REPO = "3Craft/voxkit"
BUNDLE_FILENAME = "voxkit-models.tar.gz"
BUNDLE_MANIFEST_FILENAME = "voxkit-models.manifest.json"

# ── HF token 路径（按优先级）──────────────────────────────
HF_TOKEN_PATHS: list[Path] = [
    Path.home() / ".cache" / "huggingface" / "token",
    Path.home() / ".huggingface" / "token",
]

# ── voxkit 用户级 venv ──────────────────────────────────
VENV_DIR: Path = Path.home() / ".local" / "share" / "voxkit" / "venv"
VENV_PYTHON: Path = VENV_DIR / "bin" / "python"
INSTALLED_MARKER: Path = Path.home() / ".cache" / "voxkit" / ".installed"

# pyannote.audio 版本（与 pyproject.toml [worker] 一致）
PYANNOTE_VERSION_SPEC = "pyannote.audio>=4.0.4,<5"

# Worker stdout JSON sentinel（避开 torch/pyannote 偶发 print 干扰）
WORKER_JSON_SENTINEL = "__VOXKIT_JSON__"

# whisper.cpp 在这些语言下不输出 word 级时间戳；segmenter 走 phrase 模式，
# whisper_exec 跳过 --max-len 1 / --split-on-word。集中定义避免漂移。
CJK_LANGUAGES: frozenset[str] = frozenset({"zh", "ja", "yue", "ko"})

# pyannote 偶尔吐 17ms / 80ms 的瞬时 alt-speaker burst（重叠检测 / 聚类抖动）；
# 短于此阈值的 dia 段被 align_speakers 视为噪声丢弃，避免污染 majority vote。
# 经验值 0.5s 在典型播客上稳定；调小会让短促"yeah"被归到正确说话人但风险变高。
DIA_PHANTOM_FILTER_S: float = 0.5


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
    "BUNDLE_MODELS", "BUNDLE_AUX_FILES", "BUNDLE_GITHUB_REPO",
    "BUNDLE_FILENAME", "BUNDLE_MANIFEST_FILENAME",
    "SILERO_VAD_FILENAME",
    "WHISPER_CPP_BREW_SHARE_APPLE", "WHISPER_CPP_BREW_SHARE_INTEL",
    "VOXKIT_AUX_DIR_TILDE",
    "HF_HUB_OFFLINE_ENV",
    "HF_TOKEN_PATHS",
    "VENV_DIR", "VENV_PYTHON", "INSTALLED_MARKER",
    "PYANNOTE_VERSION_SPEC",
    "WORKER_JSON_SENTINEL",
    "CJK_LANGUAGES",
    "DIA_PHANTOM_FILTER_S",
    "ExitCode",
]

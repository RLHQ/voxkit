"""voxkit setup — 显式触发 lazy install 等同流程。

逻辑等价于第一次跑 diarize 时的自动安装：
  1. 创建 ~/.local/share/voxkit/venv（uv，Python 3.12）
  2. 安装 pyannote.audio 4.x + 当前 voxkit 包（带 worker extra）
  3. 写标记文件 ~/.cache/voxkit/.installed
  4. 可选：触发模型预下载（避免首次 diarize 卡）
"""

from __future__ import annotations

from voxkit.core import lazy_install


def run() -> int:
    """显式执行 lazy install。返回 exit code。"""
    print("[setup] 检查并准备 voxkit venv...")
    try:
        info = lazy_install.ensure_venv(verbose=True)
    except lazy_install.SetupError as e:
        print(f"❌ setup 失败: {e}")
        return 1
    print(f"✅ venv 就绪: {info.venv_path}")
    print(f"   pyannote.audio: {info.pyannote_version}")
    print(f"   标记文件: {info.installed_marker}")
    print()
    print("提示：模型权重首次推理时由 HF Hub 按需下载（建议先跑一次短音频热身）。")
    return 0


__all__ = ["run"]

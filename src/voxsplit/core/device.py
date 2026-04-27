"""设备选择：MPS > CUDA > CPU。

仅在 worker 进程内调用（torch 才可用）。CLI 主进程不要 import 本模块。
"""

from __future__ import annotations

from typing import Tuple


def select_device(prefer: str = "auto") -> Tuple[object, str]:
    """根据 prefer 选择设备。返回 (torch.device, 友好名)。

    prefer ∈ {auto, mps, cuda, cpu}。auto 走 MPS > CUDA > CPU。
    显式指定但不可用 → fallback 到 auto 并附 warn 名称（调用方自行 emit warn）。
    """
    import torch

    def _mps():
        return torch.device("mps"), "mps"

    def _cuda():
        return torch.device("cuda"), "cuda"

    def _cpu():
        return torch.device("cpu"), "cpu"

    if prefer == "mps" and torch.backends.mps.is_available():
        return _mps()
    if prefer == "cuda" and torch.cuda.is_available():
        return _cuda()
    if prefer == "cpu":
        return _cpu()

    # auto / fallback
    if torch.backends.mps.is_available():
        return _mps()
    if torch.cuda.is_available():
        return _cuda()
    return _cpu()


__all__ = ["select_device"]

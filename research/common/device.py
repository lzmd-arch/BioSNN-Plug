"""设备选择与显存测量。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

计划书 §6.2 把 **8GB 显存列为硬约束**，§12.4 要求 GPU 实验记录**显存峰值**，并在
超出预算时说明是否触发了 §6.2 的降级路径。所以"用了哪块卡"和"峰值多少"不是
实验的附带信息，是结论的一部分——同一个准确率，跑在 2GB 和跑在 7.9GB 上不是一回事。

本机是 RTX 5060（8GB，sm_120）。sm_120 要求 CUDA 12.8 以上的轮子，本仓库按平台
指向 ``download.pytorch.org/whl/cu130``（见 ADR-0008 与 ``pyproject.toml``）。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

__all__ = ["DeviceInfo", "describe_device", "measure_peak_memory", "select_device"]


def _torch() -> Any:
    import torch

    return torch


def select_device(prefer: str | None = None) -> Any:
    """选一个 ``torch.device``。

    Args:
        prefer: ``"cuda"`` / ``"cpu"``；``None`` 时自动（有 CUDA 用 CUDA）。

    Returns:
        ``torch.device``。
    """
    torch = _torch()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 cuda 但当前环境不可用。本仓库的 CUDA 轮子按平台从 "
                "download.pytorch.org/whl/cu130 取（见 ADR-0008）；CI 上走的是 CPU 索引。"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DeviceInfo:
    """设备与显存信息，直接进复现记录的「硬件」一栏。"""

    kind: str  # "cuda" 或 "cpu"
    name: str
    total_memory_mb: float | None
    capability: tuple[int, int] | None

    def render(self) -> str:
        if self.kind == "cpu":
            return f"CPU（{self.name}，无 GPU 参与）"
        cap = f"sm_{self.capability[0]}{self.capability[1]}" if self.capability else "?"
        return f"{self.name}，{self.total_memory_mb:,.0f} MB，{cap}"


def describe_device(device: Any = None) -> DeviceInfo:
    """读取当前设备的可记录信息。"""
    torch = _torch()
    device = select_device() if device is None else device

    if device.type != "cuda":
        import platform

        return DeviceInfo(kind="cpu", name=platform.processor() or platform.machine(),
                          total_memory_mb=None, capability=None)

    props = torch.cuda.get_device_properties(device)
    return DeviceInfo(
        kind="cuda",
        name=props.name,
        total_memory_mb=props.total_memory / (1024**2),
        capability=torch.cuda.get_device_capability(device),
    )


@contextmanager
def measure_peak_memory(device: Any = None) -> Iterator[dict[str, float]]:
    """测一段代码的显存峰值（MiB）。

    用法::

        with measure_peak_memory(device) as stats:
            train(...)
        print(stats["peak_mb"])

    在 CPU 上 ``peak_mb`` 为 ``0.0``——不是"没测到"，是确实没有显存这回事。

    用 ``max_memory_allocated``（PyTorch 分配器口径）而不是 ``nvidia-smi``：
    后者含其它进程与缓存，会把结果抬高到无法归因。
    """
    torch = _torch()
    device = select_device() if device is None else device
    stats: dict[str, float] = {"peak_mb": 0.0}

    if device.type != "cuda":
        yield stats
        return

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        yield stats
    finally:
        torch.cuda.synchronize(device)
        stats["peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        stats["reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2)

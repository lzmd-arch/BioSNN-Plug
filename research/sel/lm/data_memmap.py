"""memmap 语料的按窗采样：**只把当前 batch 读进内存**。

审计发现的问题：`run_stream` 原先用 ``torch.from_numpy(np.array(flat[:split], dtype=int64))``，
那会把整个语料物化成 int64 —— 2.4B token 就是 **19 GB 内存**。那不是 memmap。

本模块只做一件事：给定 ``uint16`` 的 flat 语料，按随机偏移切 ``(batch, window)`` 的窗口，
**峰值内存 = batch × window × 8 字节**，与语料大小无关。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["WindowSampler"]


class WindowSampler:
    """从 ``uint16`` flat 语料里随机切窗口，返回 ``(tokens, valid)`` 的 ``int64`` 张量。"""

    def __init__(
        self,
        path: Path,
        *,
        batch: int,
        window: int,
        generator: torch.Generator,
        limit: int | None = None,
        device="cpu",
    ) -> None:
        self.flat = np.memmap(path, dtype=np.uint16, mode="r")
        self.size = int(self.flat.size if limit is None else min(limit, self.flat.size))
        self.batch = int(batch)
        self.window = int(window)
        self.generator = generator
        self.device = device
        if self.size <= self.window + 1:
            raise ValueError(f"语料太小：{self.size} <= window+1 ({self.window + 1})")

    def __len__(self) -> int:
        return self.size - self.window - 1

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """切一个 batch。窗口内全有效，所以 ``valid`` 恒为 True。"""
        high = self.size - self.window - 1
        offsets = torch.randint(0, high, (self.batch,), generator=self.generator).tolist()
        # 只有这一步把数据搬进内存：batch × window 个 uint16
        chunk = np.empty((self.batch, self.window + 1), dtype=np.int64)
        for row, start in enumerate(offsets):
            chunk[row] = self.flat[start : start + self.window + 1]
        tokens = torch.from_numpy(chunk).to(self.device)
        valid = torch.ones(self.batch, self.window + 1, dtype=torch.bool, device=self.device)
        return tokens, valid

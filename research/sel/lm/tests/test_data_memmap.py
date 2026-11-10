"""守卫：按窗采样器不物化整个语料，且切出来的窗口与 memmap 一致。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from research.sel.lm.data_memmap import WindowSampler


def _corpus(tmp_path: Path, n: int = 5000) -> Path:
    path = tmp_path / "c.u16"
    np.arange(n, dtype=np.uint16).tofile(path)
    return path


def test_windows_match_the_file(tmp_path: Path) -> None:
    path = _corpus(tmp_path)
    sampler = WindowSampler(path, batch=4, window=32, generator=torch.Generator().manual_seed(0))
    tokens, valid = sampler.next()
    assert tokens.shape == (4, 33)
    assert int(tokens.max()) < 5000
    # 每个窗口必须是**连续递增**的（因为我们写的就是 0,1,2,…）
    diffs = tokens[:, 1:] - tokens[:, :-1]
    assert bool((diffs == 1).all()), "窗口不是连续切片"
    assert bool(valid.all())


def test_does_not_materialise_the_corpus(tmp_path: Path) -> None:
    """采样器只持有 memmap，不应把语料复制进 ndarray。"""
    path = _corpus(tmp_path, n=200_000)
    sampler = WindowSampler(path, batch=2, window=16, generator=torch.Generator().manual_seed(0))
    assert isinstance(sampler.flat, np.memmap), "底层应该是 memmap 而不是 ndarray"
    assert sampler.size == 200_000


def test_rejects_too_small_corpus(tmp_path: Path) -> None:
    path = _corpus(tmp_path, n=10)
    with pytest.raises(ValueError):
        WindowSampler(path, batch=2, window=32, generator=torch.Generator().manual_seed(0))

"""守卫：融合核与 PyTorch 参考路径**逐项一致**。

融合核改的是访存方式，不是数学。所以这一条的容差要被 fp32 的求和顺序解释掉
（`tl.sum` 的归约顺序与 cuBLAS 不同，1e-5 量级是正常的）。
"""

from __future__ import annotations

import pytest
import torch

from research.sel.lm import tree_kernel
from research.sel.lm.hierarchical import build_huffman_paths, path_deltas, path_logits

pytestmark = pytest.mark.skipif(
    not (tree_kernel.HAS_TRITON and torch.cuda.is_available()),
    reason="需要 Triton + CUDA",
)


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    counts = torch.tensor([20.0, 9.0, 5.0, 3.0, 2.0, 1.0, 1.0, 1.0])
    paths = build_huffman_paths(counts).to("cuda")
    weight = (torch.randn(paths.n_nodes, 96, device="cuda") * 0.3).contiguous()
    features = torch.randn(5, 96, device="cuda")
    targets = torch.tensor([0, 3, 7, 2, 5], device="cuda")
    return paths, weight, features, targets


def test_gather_dot_matches_reference() -> None:
    paths, weight, features, targets = _setup()
    logits_ref, valid, safe = path_logits(features, weight, paths, targets, bias=None)
    out = tree_kernel.gather_dot(features, weight, safe)
    out = torch.where(valid, out, torch.zeros_like(out))
    assert torch.allclose(out, logits_ref, atol=2e-4, rtol=2e-4), (out - logits_ref).abs().max()


def test_gather_signal_matches_reference() -> None:
    paths, weight, features, targets = _setup(1)
    logits, valid, safe = path_logits(features, weight, paths, targets, bias=None)
    bits = paths.bits.index_select(0, targets).to(logits.dtype)
    delta = path_deltas(torch.sigmoid(logits), bits, valid)
    ref = torch.zeros_like(features)
    for j in range(safe.shape[1]):
        ref.add_(weight.index_select(0, safe[:, j]) * delta[:, j].unsqueeze(1))
    out = tree_kernel.gather_signal(delta.contiguous(), weight, safe)
    assert torch.allclose(out, ref, atol=2e-4, rtol=2e-4), (out - ref).abs().max()


def test_signal_matches_full_step_signal() -> None:
    """端到端：核算出的信号与 `TreeReadout.step(update=False)` 返回的信号同值。"""
    from research.sel.lm.hierarchical import TreeReadout

    paths, _weight, features, targets = _setup(2)
    readout = TreeReadout(paths, 96, generator=torch.Generator().manual_seed(2), device="cuda")
    _, signal_ref = readout.step(features, targets, update=False)

    logits, valid, safe = path_logits(features, readout.weight, paths, targets, readout.bias)
    bits = paths.bits.index_select(0, targets).to(logits.dtype)
    delta = path_deltas(torch.sigmoid(logits), bits, valid)
    signal = tree_kernel.gather_signal(delta.contiguous(), readout.weight, safe)
    assert torch.allclose(signal, signal_ref, atol=5e-4, rtol=5e-4), (
        (signal - signal_ref).abs().max()
    )


def test_readout_kernel_and_pytorch_paths_agree() -> None:
    """整步对齐：核路径与 PyTorch 路径在**同样的初始权重**上给出同样的损失、信号与更新。"""
    from research.sel.lm.hierarchical import TreeReadout

    paths, _weight, features, targets = _setup(3)
    ker = TreeReadout(paths, 96, generator=torch.Generator().manual_seed(3), device="cuda")
    ref = TreeReadout(paths, 96, generator=torch.Generator().manual_seed(3), device="cuda")
    ref.use_kernel = False
    assert ker.use_kernel
    assert torch.equal(ker.weight, ref.weight) and torch.equal(ker.bias, ref.bias)

    for _ in range(3):
        nll_k, sig_k = ker.step(features, targets)
        nll_r, sig_r = ref.step(features, targets)
    assert torch.allclose(nll_k, nll_r, atol=1e-3)
    assert torch.allclose(sig_k, sig_r, atol=1e-3), (sig_k - sig_r).abs().max()
    assert torch.allclose(ker.weight, ref.weight, atol=1e-3), (ker.weight - ref.weight).abs().max()

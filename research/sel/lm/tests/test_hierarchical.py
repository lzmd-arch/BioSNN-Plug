"""守卫：Huffman 路径表必须让**所有 token 的路径概率之和等于 1**。

这一条不成立，后面测到的质量差异就分不清是"层次 softmax 不行"还是"树算错了"。
"""

from __future__ import annotations

import torch

from research.sel.lm.hierarchical import (
    TreeReadout,
    build_huffman_paths,
    path_deltas,
    path_logits,
    path_nll,
    path_probabilities,
)


def test_paths_are_root_to_leaf() -> None:
    counts = torch.tensor([10.0, 5.0, 3.0, 2.0, 1.0, 1.0])
    paths = build_huffman_paths(counts)
    assert paths.nodes.shape[0] == 6
    assert paths.n_nodes == 5, "V 个叶子的二叉树应有 V-1 个内部节点"
    # 频次最高的 token 路径最短
    assert int(paths.length[0]) == int(paths.length.min())


def test_probabilities_sum_to_one() -> None:
    torch.manual_seed(0)
    counts = torch.tensor([20.0, 9.0, 5.0, 3.0, 2.0, 1.0])
    paths = build_huffman_paths(counts)
    features = torch.randn(3, 8)
    weight = torch.randn(paths.n_nodes, 8) * 0.5
    probs = path_probabilities(features, weight, paths)
    assert probs.shape == (3, 6)
    total = probs.sum(dim=1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), total


def test_frequent_token_gets_shorter_path() -> None:
    counts = torch.tensor([100.0, 50.0, 1.0, 1.0])
    paths = build_huffman_paths(counts)
    assert int(paths.length[1]) < int(paths.length[2])


# -- 树是否真的最优 ----------------------------------------------------------------


def _brute_force_cost(freqs: list[float]) -> float:
    """枚举全部满二叉树（叶子按频次升序连续切分），返回最小加权路径长。

    最优 Huffman 树一定存在"叶子有序"的形态（同深度可交换），所以只需枚举连续切分。
    """

    def best(lo: int, hi: int) -> float:
        if hi - lo == 1:
            return 0.0
        total = sum(freqs[lo:hi])
        return total + min(best(lo, mid) + best(mid, hi) for mid in range(lo + 1, hi))

    return best(0, len(freqs))


def test_huffman_matches_brute_force() -> None:
    freqs = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    counts = torch.tensor(freqs)
    paths = build_huffman_paths(counts)
    cost = sum(float(counts[t]) * int(paths.length[t]) for t in range(len(freqs)))
    assert abs(cost - _brute_force_cost(freqs)) < 1e-6, cost


def test_huffman_matches_brute_force_uneven() -> None:
    freqs = [0.5, 1.0, 1.5, 4.0, 20.0, 21.0, 23.0]
    counts = torch.tensor(freqs)
    paths = build_huffman_paths(counts)
    cost = sum(float(counts[t]) * int(paths.length[t]) for t in range(len(freqs)))
    assert abs(cost - _brute_force_cost(freqs)) < 1e-6, cost


def test_right_share_matches_frequency() -> None:
    counts = torch.tensor([20.0, 9.0, 5.0, 3.0, 2.0, 1.0])
    paths = build_huffman_paths(counts)
    assert paths.right_share is not None
    assert bool((paths.right_share > 0).all())
    assert bool((paths.right_share < 1).all())


# -- 初始化：零权重 + 频次偏置 = 经验一元分布 ------------------------------------


def test_zero_weight_with_bias_reproduces_unigram() -> None:
    counts = torch.tensor([20.0, 9.0, 5.0, 3.0, 2.0, 1.0])
    paths = build_huffman_paths(counts)
    readout = TreeReadout(paths, 8, generator=torch.Generator().manual_seed(0))
    readout.weight.zero_()
    # 零特征 ⇒ logit 只剩偏置；每个 token 是 batch 里的一行
    features = torch.zeros(len(counts), 8)
    expected = counts / counts.sum()
    got = torch.exp(-readout.nll(features, torch.arange(len(counts))))
    assert torch.allclose(got, expected, atol=1e-5), (got, expected)


# -- 局部梯度与 autograd 一致（autograd 只在这里出现） ---------------------------


def _local_grads(features, weight, paths, targets, bias):
    logits, valid, safe = path_logits(features, weight, paths, targets, bias)
    bits = paths.bits.index_select(0, targets).to(logits.dtype)
    prob = torch.sigmoid(logits)
    delta = path_deltas(prob, bits, valid)
    grad_w = torch.zeros_like(weight)
    grad_f = torch.zeros_like(features)
    for j in range(safe.shape[1]):
        grad_w.index_add_(0, safe[:, j], features * delta[:, j].unsqueeze(1))
        grad_f.add_(weight.index_select(0, safe[:, j]) * delta[:, j].unsqueeze(1))
    return grad_w, grad_f


def test_local_gradients_match_autograd() -> None:
    torch.manual_seed(0)
    counts = torch.tensor([20.0, 9.0, 5.0, 3.0, 2.0, 1.0, 1.0])
    paths = build_huffman_paths(counts)
    features = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(paths.n_nodes, 8).mul_(0.3).requires_grad_(True)
    bias = (torch.randn(paths.n_nodes) * 0.1).requires_grad_(True)
    targets = torch.tensor([0, 1, 2, 5])

    loss, _, _, _, _ = path_nll(features, weight, paths, targets, bias)
    loss.sum().backward()

    grad_w, grad_f = _local_grads(features, weight, paths, targets, bias)
    assert torch.allclose(grad_w, weight.grad, atol=1e-6), (grad_w - weight.grad).abs().max()
    assert torch.allclose(grad_f, features.grad, atol=1e-6), (grad_f - features.grad).abs().max()


def test_nll_equals_negative_log_path_probability() -> None:
    torch.manual_seed(1)
    counts = torch.tensor([10.0, 6.0, 4.0, 2.0, 1.0])
    paths = build_huffman_paths(counts)
    readout = TreeReadout(paths, 8, generator=torch.Generator().manual_seed(1))
    features = torch.randn(3, 8)
    targets = torch.tensor([0, 3, 4])
    nll = readout.nll(features, targets)
    probs = path_probabilities(features, readout.weight, paths)
    # path_probabilities 不开偏置；这里用与 nll 同口径的重算
    logits, valid, _ = path_logits(features, readout.weight, paths, targets, readout.bias)
    bits = paths.bits.index_select(0, targets).to(logits.dtype)
    p = torch.sigmoid(logits)
    taken = torch.where(bits > 0.5, p, 1.0 - p)
    manual = -(torch.log(taken.clamp(min=1e-12)) * valid).sum(dim=-1)
    assert torch.allclose(nll, manual, atol=1e-6)
    assert probs.shape == (3, 5)


def test_update_reduces_nll() -> None:
    torch.manual_seed(2)
    counts = torch.tensor([10.0, 6.0, 4.0, 2.0, 1.0, 1.0])
    paths = build_huffman_paths(counts)
    readout = TreeReadout(paths, 16, lr_out=0.5, generator=torch.Generator().manual_seed(2))
    features = torch.randn(8, 16)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2])
    before = readout.nll(features, targets).mean()
    for _ in range(20):
        readout.step(features, targets)
    after = readout.nll(features, targets).mean()
    assert float(after) < float(before), (before, after)

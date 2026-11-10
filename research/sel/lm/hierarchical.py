"""层次 softmax 的词典树、路径概率与**局部梯度**（`prereg-lm-hierarchical-softmax.md`）。

三块依次是：**树**（`build_huffman_paths`）、**概率**（`path_probabilities`）、
**路径上的二分类损失与局部更新**（`TreeReadout`）。

为什么先把树钉死：H-0 已实测通过（按行主序取 `(V, M)` 权重时三条路径都 ~7x），
但**概率是否正确**必须先在树这一层验证 —— 否则后面测到的质量差异分不清是
"层次 softmax 不行"还是"树算错了"。

守卫（`tests/test_hierarchical.py`）：

1. **所有 token 的路径概率之和等于 1**；
2. **树是真正的最优 Huffman 树**（V=6 时与暴力枚举的最小加权路径长比对）；
3. **局部梯度与 autograd 逐项一致** —— autograd 只出现在测试里，训练路径上不出现；
4. **零权重 + 频次偏置**时树恰好复现经验一元分布。
"""

from __future__ import annotations

import heapq
import math

import torch

__all__ = [
    "HuffmanPaths",
    "TreeReadout",
    "build_huffman_paths",
    "path_deltas",
    "path_logits",
    "path_nll",
    "path_probabilities",
]


class HuffmanPaths:
    """每个 token 的根到叶路径。

    ``nodes[t, j]`` 是第 j 层的内部节点 id（-1 表示该 token 路径更短，补位）；
    ``bits[t, j]`` 是走向（1 = 右子）。``length[t]`` 是真实深度。
    ``right_share[n]`` 是内部节点 n **按训练频次**走向右子的概率 —— 只用于初始化偏置。
    """

    def __init__(
        self,
        nodes: torch.Tensor,
        bits: torch.Tensor,
        length: torch.Tensor,
        n_nodes: int,
        right_share: torch.Tensor | None = None,
    ):
        self.nodes = nodes
        self.bits = bits
        self.length = length
        self.n_nodes = n_nodes
        self.right_share = right_share

    def __len__(self) -> int:
        return self.nodes.shape[0]

    @property
    def depth(self) -> int:
        return int(self.nodes.shape[1])


def build_huffman_paths(counts: torch.Tensor) -> HuffmanPaths:
    """按 token 频次建 Huffman 树，返回路径表。频次越高路径越短（经典做法）。

    **频次为 0 的叶子不许剔除**：树里的叶子必须覆盖全部词表 id，否则索引对不上。
    它们会被压到最深处，这是正确的行为（生僻字本来就该贵）。
    """
    vocab = int(counts.numel())
    if vocab < 2:
        raise ValueError("词表至少 2 个 token")
    freq = [float(counts[i].item()) for i in range(vocab)]
    heap = [(freq[i], i, ("leaf", i)) for i in range(vocab)]
    heapq.heapify(heap)
    children: dict[int, tuple[tuple[str, int], tuple[str, int]]] = {}
    totals: dict[tuple[str, int], float] = {("leaf", i): freq[i] for i in range(vocab)}
    next_id = 0
    while len(heap) > 1:
        wl, _, left = heapq.heappop(heap)
        wr, _, right = heapq.heappop(heap)
        node = next_id  # 内部节点 id 与 token id 分开编号
        children[node] = (left, right)
        # **子树频次必须累加**：早期版本只在两侧都是叶子时取频次、否则记 0，
        # 于是合并出来的内部节点权重丢失，建出的树只是一棵合法的二叉树、不是 Huffman 树。
        # `test_huffman_matches_brute_force` 就是为这个 bug 设的。
        totals[("node", node)] = wl + wr
        heapq.heappush(heap, (wl + wr, node, ("node", node)))
        next_id += 1

    root = heap[0][2]
    paths: list[list[tuple[int, int]]] = [[] for _ in range(vocab)]

    def walk(node: tuple[str, int], trail: list[tuple[int, int]]) -> None:
        kind, value = node
        if kind == "leaf":
            paths[value] = list(trail)
            return
        left, right = children[value]
        walk(left, [*trail, (value, 0)])
        walk(right, [*trail, (value, 1)])

    walk(root, [])
    depth = max(len(p) for p in paths)
    nodes = torch.full((vocab, depth), -1, dtype=torch.long)
    bits = torch.zeros(vocab, depth, dtype=torch.long)
    length = torch.tensor([len(p) for p in paths], dtype=torch.long)
    for t, path in enumerate(paths):
        for j, (node, bit) in enumerate(path):
            nodes[t, j] = node
            bits[t, j] = bit

    share = torch.full((next_id,), 0.5, dtype=torch.float32)
    for node, pair in children.items():
        total = totals[("node", node)]
        if total > 0:
            share[node] = totals[pair[1]] / total
    return HuffmanPaths(nodes, bits, length, next_id, share)


def path_probabilities(
    features: torch.Tensor, weight: torch.Tensor, paths: HuffmanPaths
) -> torch.Tensor:
    """返回 ``(B, V)`` 的路径概率；对每个 token 是路径上各节点二分类概率之积。

    实现按 H-0 的口径：**权重存 ``(n_nodes, M)``，按行取**（连续行，带宽友好）。
    """
    depth = paths.nodes.shape[1]
    batch = features.shape[0]
    logp = torch.zeros(batch, len(paths), dtype=features.dtype, device=features.device)
    for j in range(depth):
        ids = paths.nodes[:, j]
        valid = ids >= 0
        if not bool(valid.any()):
            continue
        node_ids = ids[valid]
        targets = paths.bits[valid, j].to(features.dtype)
        logits = torch.einsum("bm,km->bk", features, weight[node_ids])
        prob = torch.sigmoid(logits)
        prob = torch.where(targets.unsqueeze(0) > 0.5, prob, 1.0 - prob)
        logp[:, valid] += torch.log(prob.clamp(min=1e-12))
    return logp.exp()


def _gather_path_nodes(paths: HuffmanPaths, targets: torch.Tensor):
    """``(safe_ids, valid)``，两者都是 ``(B, depth)``。

    ``-1`` 补位被夹到 0，对应位置的输出随后按 ``valid`` 清零，所以夹取不改变任何数值。
    """
    ids = paths.nodes.index_select(0, targets)
    valid = ids >= 0
    return ids.clamp(min=0), valid


def path_logits(
    features: torch.Tensor,
    weight: torch.Tensor,
    paths: HuffmanPaths,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    """路径上每个节点的 logit。返回 ``(logits (B, depth), valid (B, depth), safe_ids)``。"""
    safe, valid = _gather_path_nodes(paths, targets)
    flat = safe.reshape(-1)
    gathered = weight.index_select(0, flat).view(*safe.shape, weight.shape[1])
    logits = torch.einsum("bm,bkm->bk", features, gathered)
    if bias is not None:
        logits = logits + bias.index_select(0, flat).view(safe.shape)
    logits = torch.where(valid, logits, torch.zeros_like(logits))
    return logits, valid, safe


def path_nll(
    features: torch.Tensor,
    weight: torch.Tensor,
    paths: HuffmanPaths,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    """逐样本的负对数似然 ``(B,)``，以及 ``(prob, bits, valid, safe_ids)``。

    ``prob`` 是**节点上「走右子」的概率**（不是走了哪一边之后的概率），梯度要用它。
    """
    logits, valid, safe = path_logits(features, weight, paths, targets, bias)
    bits = paths.bits.index_select(0, targets).to(logits.dtype)
    prob = torch.sigmoid(logits)
    taken = torch.where(bits > 0.5, prob, 1.0 - prob)
    logp = torch.log(taken.clamp(min=1e-12)) * valid
    return -logp.sum(dim=-1), prob, bits, valid, safe


def path_deltas(prob: torch.Tensor, bits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``dL/dlogit = p - bit``（BCE 的经典结果），补位处为 0。

    推导：``P = p`` 当 ``bit=1``、``P = 1-p`` 当 ``bit=0``；``d(-logP)/dp`` 与
    ``dp/dlogit = p(1-p)`` 相乘后两种情形都恰好是 ``p - bit``。
    """
    return torch.where(valid, prob - bits, torch.zeros_like(prob))


class TreeReadout:
    """树读出的**局部更新**：每个内部节点的权重只按自己的二分类标签改。

    ``weight`` 形状 ``(n_nodes, M)``，**行主序** —— H-0 实测的唯一快布局。
    更新走 ``index_add_``，信号回到 ``features`` 的路径与稠密版
    ``gradient @ w_out`` 同构（都是"输出侧误差乘回权重"），**不需要反向传播**。

    ``bias``（``n_nodes`` 个标量）不是可选项，是为了**初始化公平**：零权重时
    每个节点都是 0.5，于是每个 token 的概率是 ``2^-深度``，高频短路径词的先验
    会远离一元分布。按 ``right_share`` 取 log-odds 后，零权重处恰好复现经验一元分布。
    """

    def __init__(
        self,
        paths: HuffmanPaths,
        n_features: int,
        *,
        lr_out: float = 0.5,
        bias_init: bool = True,
        generator: torch.Generator | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.paths = paths
        self.n_features = int(n_features)
        self.lr_out = float(lr_out)
        gen = generator or torch.Generator()
        weight = torch.randn(paths.n_nodes, n_features, generator=gen) / math.sqrt(n_features)
        self.weight = weight.to(device=device, dtype=dtype)
        if bias_init and paths.right_share is not None:
            share = paths.right_share.to(device=device, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            self.bias = torch.log(share / (1.0 - share)).to(dtype)
        else:
            self.bias = torch.zeros(paths.n_nodes, device=device, dtype=dtype)

    @property
    def n_nodes(self) -> int:
        return self.paths.n_nodes

    def nll(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """逐样本 NLL ``(B,)``，不更新。"""
        out, _, _, _, _ = path_nll(features, self.weight, self.paths, targets, self.bias)
        return out

    def step(self, features: torch.Tensor, targets: torch.Tensor, *, update: bool = True):
        """一步：返回 ``(nll (B,), signal (B, M))``。

        返回的 ``signal`` 就是稠密版里 ``gradient @ w_out`` 的角色 —— 交给 e-prop
        的 ψ 与资格痕迹继续往下传。**这一路没有 autograd。**
        """
        nll, prob, bits, valid, safe = path_nll(
            features, self.weight, self.paths, targets, self.bias
        )
        delta = path_deltas(prob, bits, valid)
        depth = int(safe.shape[1])
        signal = torch.zeros_like(features)
        for j in range(depth):
            signal.add_(self.weight.index_select(0, safe[:, j]) * delta[:, j].unsqueeze(1))
        if not update:
            return nll, signal

        batch = max(int(features.shape[0]), 1)
        denom = features.pow(2).sum(dim=1).mean().clamp(min=1e-3)
        # 逐层 index_add_：峰值内存是 (B, M) 而不是 (B·depth, M)。
        # B=256、depth=12、M=16384 时那是 16 MB 对 201 MB。
        for j in range(depth):
            scale = (delta[:, j] / (denom * batch)).unsqueeze(1)
            self.weight.index_add_(0, safe[:, j], features * scale, alpha=-self.lr_out)
            self.bias.index_add_(0, safe[:, j], scale.squeeze(1), alpha=-self.lr_out)
        return nll, signal

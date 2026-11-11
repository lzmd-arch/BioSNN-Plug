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

from research.sel.lm import tree_kernel

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

    def to(self, device) -> HuffmanPaths:
        """建树在 CPU（heapq 是纯 Python），进模型前要搬到与权重同一个设备。"""
        self.nodes = self.nodes.to(device)
        self.bits = self.bits.to(device)
        self.length = self.length.to(device)
        if self.right_share is not None:
            self.right_share = self.right_share.to(device)
        return self


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
    features: torch.Tensor,
    weight: torch.Tensor,
    paths: HuffmanPaths,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 ``(B, V)`` 的路径概率；对每个 token 是路径上各节点二分类概率之积。

    实现按 H-0 的口径：**权重存 ``(n_nodes, M)``，按行取**（连续行，带宽友好）。
    ``bias`` 必须与训练时用的是同一个，否则评测的分布与训练的不是同一个。
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
        if bias is not None:
            logits = logits + bias[node_ids]
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

    **偏置默认冻结**（``bias_train=False``）。它是从训练频次算出来的**先验**，与树结构
    同一个地位，不是学出来的参数。这不是保守，是实测：让偏置跟着权重一起走同一个
    NLMS 步长时它会发散 —— 偏置没有 ``features`` 乘着，`1/norm` 这个归一化对它毫无意义，
    实测 600 步内跑到 ``[-29.6, +99.3]``（``sigmoid`` 全程饱和），训练 CE 0.41 而
    评测 CE 11.3。稠密读出本来就不带偏置，冻结偏置让两边"学的部分"都是纯权重。
    """

    def __init__(
        self,
        paths: HuffmanPaths,
        n_features: int,
        *,
        lr_out: float = 0.5,
        bias_init: bool = True,
        bias_train: bool = False,
        use_kernel: bool | None = None,
        packed: bool = False,
        generator: torch.Generator | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.paths = paths
        self.n_features = int(n_features)
        self.lr_out = float(lr_out)
        self.bias_train = bool(bias_train)
        gen = generator or torch.Generator()
        weight = torch.randn(paths.n_nodes, n_features, generator=gen) / math.sqrt(n_features)
        self.weight = weight.to(device=device, dtype=dtype)
        #: 融合核把「gather + 点积」「gather + 信号」各压成**一次** DRAM 读。
        #: 没有 CUDA / Triton 时退回 PyTorch 路径；两条路在 `test_tree_kernel.py` 里逐项对齐。
        self.use_kernel = (
            tree_kernel.available(self.weight.device) if use_kernel is None else bool(use_kernel)
        )
        #: ragged 布局：只走真实存在的 (b, j) 对。**默认关**——实测（V=32,000、B=256、M=4,096）
        #: 打包本身的开销（8 个小内核 + 布尔索引）比省下的访存还贵：整步 2.23 → 2.77 ms。
        #: 真正吃掉补位的是核里的 `if v < 0: return`（早退），那条路默认就走。
        self.packed = bool(packed)
        if bias_init and paths.right_share is not None:
            share = paths.right_share.to(device=device, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            self.bias = torch.log(share / (1.0 - share)).to(dtype)
        else:
            self.bias = torch.zeros(paths.n_nodes, device=device, dtype=dtype)

    @property
    def n_nodes(self) -> int:
        return self.paths.n_nodes

    def _path_logits(self, features: torch.Tensor, targets: torch.Tensor):
        """``(logits, valid, safe_ids)``；有融合核就走核，否则走 PyTorch。"""
        if not self.use_kernel:
            return path_logits(features, self.weight, self.paths, targets, self.bias)
        # 传**未夹取**的 ids：补位是 -1，核里直接早退，一行都不读
        ids = self.paths.nodes.index_select(0, targets)
        valid = ids >= 0
        safe = ids.clamp(min=0)
        logits = tree_kernel.gather_dot(features, self.weight, ids)
        if self.bias is not None:
            logits = logits + self.bias.index_select(0, safe.reshape(-1)).view(safe.shape)
        logits = torch.where(valid, logits, torch.zeros_like(logits))
        return logits, valid, safe

    def _step_packed(self, features: torch.Tensor, targets: torch.Tensor, *, update, mask):
        """ragged 版的一步：只处理真实配对，不碰补位。"""
        flat, rows, bits, offs = tree_kernel.pack_paths(self.paths, targets)
        logits = tree_kernel.packed_dot(features, self.weight, flat, rows)
        if self.bias is not None:
            logits = logits + self.bias.index_select(0, flat)
        prob = torch.sigmoid(logits)
        bits_f = bits.to(logits.dtype)
        taken = torch.where(bits_f > 0.5, prob, 1.0 - prob)
        logp = -torch.log(taken.clamp(min=1e-12))
        # **信号必须与稠密同口径除以 batch**：稠密的 `gradient` 在算信号**之前**就除了 batch，
        # 这里漏掉的话回给隐层的信号会大 B 倍（B=16 时实测隐层发放率 0.0424 → 0.1439，
        # top-1 掉 3.8pp）。H-1 那一臂就是踩了这个。
        batch = int(features.shape[0])
        delta = (prob - bits_f) / max(batch, 1)
        if mask is not None:
            keep = mask.index_select(0, rows)
            logp = logp * keep
            delta = torch.where(keep, delta, torch.zeros_like(delta))
        nll = torch.zeros(batch, dtype=logp.dtype, device=logp.device).index_add_(0, rows, logp)
        signal = tree_kernel.packed_signal(delta, self.weight, flat, offs, batch)
        signal = signal.to(features.dtype)
        if update:
            denom = features.pow(2).sum(dim=1).mean().clamp(min=1e-3)
            scale = delta / denom
            self.weight.index_add_(
                0, flat, features.index_select(0, rows) * scale.unsqueeze(1), alpha=-self.lr_out
            )
        return nll, signal

    def nll(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """逐样本 NLL ``(B,)``，不更新。"""
        out, _, _, _, _ = path_nll(features, self.weight, self.paths, targets, self.bias)
        return out

    def step(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        *,
        update: bool = True,
        mask: torch.Tensor | None = None,
    ):
        """一步：返回 ``(nll (B,), signal (B, M))``。

        返回的 ``signal`` 就是稠密版里 ``gradient @ w_out`` 的角色 —— 交给 e-prop
        的 ψ 与资格痕迹继续往下传。**这一路没有 autograd。**

        ``mask``（``(B,)`` 的 bool）为 False 的行不参与损失、信号与更新，
        与稠密版 ``gradient = gradient * active`` 同口径。
        """
        if self.use_kernel and self.packed:
            return self._step_packed(features, targets, update=update, mask=mask)
        logits, valid, safe = self._path_logits(features, targets)
        bits = self.paths.bits.index_select(0, targets).to(logits.dtype)
        prob = torch.sigmoid(logits)
        taken = torch.where(bits > 0.5, prob, 1.0 - prob)
        nll = -(torch.log(taken.clamp(min=1e-12)) * valid).sum(dim=-1)
        if mask is not None:
            valid = valid & mask.unsqueeze(1)
            nll = torch.where(mask, nll, torch.zeros_like(nll))
        # 与稠密同口径：信号、更新共用一个已经除以 batch 的 delta
        delta = path_deltas(prob, bits, valid) / max(int(features.shape[0]), 1)
        batch = int(features.shape[0])
        # **扁平实现**：把 (B·depth) 个节点摊成一次 index_select / 一次 index_add。
        # 逐层写（`for j in range(depth)`）会调 depth 次小内核，实测在 V=3,001 上是
        # 0.22x（比稠密慢 4.6 倍）；扁平化之后同一形状 0.92x、V=32,000 上 6.5x。
        # 峰值内存是 (B·depth, M) 的临时张量——1B 形状（256×10.6×16,384×4B）约 177 MB。
        flat = safe.reshape(-1)
        rows = torch.arange(batch, device=features.device).repeat_interleave(safe.shape[1])
        delta_flat = delta.reshape(-1)
        if self.use_kernel:
            signal = tree_kernel.gather_signal(delta, self.weight, safe).to(features.dtype)
        else:
            signal = torch.zeros_like(features)
            signal.index_add_(0, rows, self.weight.index_select(0, flat) * delta_flat.unsqueeze(1))
        if not update:
            return nll, signal

        denom = features.pow(2).sum(dim=1).mean().clamp(min=1e-3)
        scale = delta_flat / denom
        self.weight.index_add_(
            0, flat, features.index_select(0, rows) * scale.unsqueeze(1), alpha=-self.lr_out
        )
        if self.bias_train:
            self.bias.index_add_(0, flat, scale, alpha=-self.lr_out)
        return nll, signal

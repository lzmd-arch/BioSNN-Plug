"""层次 softmax 的词典树与路径表（`prereg-lm-hierarchical-softmax.md` 的第一块）。

为什么先只做树：H-0 已实测通过（按行主序取 `(V, M)` 权重时三条路径都 ~7x），
但**概率是否正确**必须先在树这一层钉死 —— 否则后面测到的质量差异分不清是
"层次 softmax 不行"还是"树算错了"。

守卫：**所有 token 的路径概率之和必须等于 1**（`test_hierarchical.py`）。
"""

from __future__ import annotations

import heapq

import torch

__all__ = ["HuffmanPaths", "build_huffman_paths"]


class HuffmanPaths:
    """每个 token 的根到叶路径。

    ``nodes[t, j]`` 是第 j 层的内部节点 id（-1 表示该 token 路径更短，补位）；
    ``bits[t, j]`` 是走向（1 = 右子）。``length[t]`` 是真实深度。
    """

    def __init__(self, nodes: torch.Tensor, bits: torch.Tensor, length: torch.Tensor, n_nodes: int):
        self.nodes = nodes
        self.bits = bits
        self.length = length
        self.n_nodes = n_nodes

    def __len__(self) -> int:
        return self.nodes.shape[0]


def build_huffman_paths(counts: torch.Tensor) -> HuffmanPaths:
    """按 token 频次建 Huffman 树，返回路径表。频次越高路径越短（经典做法）。"""
    vocab = int(counts.numel())
    if vocab < 2:
        raise ValueError("词表至少 2 个 token")
    heap = [(float(counts[i].item()), i, ("leaf", i)) for i in range(vocab)]
    heapq.heapify(heap)
    children: dict[int, tuple[int, int]] = {}
    next_id = 0
    while len(heap) > 1:
        _, _, left = heapq.heappop(heap)
        _, _, right = heapq.heappop(heap)
        node = next_id  # 内部节点 id 与 token id 分开编号
        children[node] = (left, right)
        weight = 0.0
        for side in (left, right):
            weight += float(counts[side[1]].item()) if side[0] == "leaf" else 0.0
        heapq.heappush(heap, (weight, node, ("node", node)))
        next_id += 1

    root = heap[0][2]
    paths: list[list[tuple[int, int]]] = [[] for _ in range(vocab)]

    def walk(node, trail: list[tuple[int, int]]) -> None:
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
    return HuffmanPaths(nodes, bits, length, next_id)


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

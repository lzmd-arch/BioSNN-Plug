"""守卫：Huffman 路径表必须让**所有 token 的路径概率之和等于 1**。

这一条不成立，后面测到的质量差异就分不清是"层次 softmax 不行"还是"树算错了"。
"""

from __future__ import annotations

import torch

from research.sel.lm.hierarchical import build_huffman_paths, path_probabilities


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

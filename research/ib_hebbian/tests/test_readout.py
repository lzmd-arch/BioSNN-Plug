"""``research.ib_hebbian.readout`` 与它的接线方式的测试。

这里盯的是两件事：闭式解**确实**是岭目标的驻点（不是"跑起来没报错"），以及
``train_mnist`` 造累积器时**不把 ``--readout-ridge`` 喂给构造函数**——那条路径曾经
在默认参数下当场崩掉，见 ``TestRidgeAccumulatorWiring``。
"""

from __future__ import annotations

import pytest
import torch

from research.ib_hebbian.readout import RidgeReadout
from research.ib_hebbian.train_mnist import build_parser, build_ridge_readout


class TestRidgeReadout:
    def test_solution_is_a_stationary_point_of_the_ridge_objective(self):
        """解必须是岭目标的驻点。

        **刻意不把 ``solve()`` 的实现抄一遍做对照**（那是同义反复）。这里用解析梯度核对：
        ``J(W) = ‖XW − Y‖²/n + λ‖W[:-1]‖²``（偏置那一行不正则化），
        ``∇J = 2/n·Xᵀ(XW − Y) + 2λ·[W[:-1]; 0]``，在最优点上应为零。
        """
        torch.manual_seed(0)
        width, n_classes, n_samples, ridge = 7, 3, 200, 1e-2
        features = torch.randn(n_samples, width)
        labels = torch.randint(0, n_classes, (n_samples,))

        accumulator = RidgeReadout(width, n_classes, ridge=ridge)
        accumulator.accumulate(features, labels)
        weight, bias = accumulator.solve()

        augmented = torch.cat(
            [features.to(torch.float64), torch.ones(n_samples, 1, dtype=torch.float64)], dim=1
        )
        onehot = torch.nn.functional.one_hot(labels, n_classes).to(torch.float64)
        solution = torch.cat([weight.to(torch.float64).T, bias.to(torch.float64).unsqueeze(0)])

        gradient = 2 * augmented.T @ (augmented @ solution - onehot) / n_samples
        gradient[:-1] += 2 * ridge * solution[:-1]
        assert gradient.abs().max().item() < 1e-7

    def test_bias_is_not_regularised(self):
        """λ 收缩的是权重，不是截距：特征恒为零时，偏置必须等于类别频率、与 λ 无关。"""
        n_samples, n_classes = 400, 2
        features = torch.zeros(n_samples, 5)
        labels = torch.cat([torch.zeros(300, dtype=torch.long), torch.ones(100, dtype=torch.long)])

        accumulator = RidgeReadout(5, n_classes, ridge=1e6)
        accumulator.accumulate(features, labels)
        weight, bias = accumulator.solve()

        assert weight.abs().max().item() < 1e-5  # 大 λ 把权重压没了
        assert bias.tolist() == pytest.approx([0.75, 0.25])  # 偏置照旧，没被压

    def test_learns_a_linearly_separated_labelling(self):
        """基本功能：线性可分的标签应当被完全拟合（这里只验证机制，不当验收数字）。"""
        torch.manual_seed(1)
        width, n_classes, n_samples = 6, 3, 300
        centres = torch.randn(n_classes, width) * 4
        labels = torch.randint(0, n_classes, (n_samples,))
        features = centres[labels] + torch.randn(n_samples, width)

        accumulator = RidgeReadout(width, n_classes, ridge=1e-3)
        accumulator.accumulate(features, labels)
        weight, bias = accumulator.solve()
        predicted = (features @ weight.T + bias).argmax(dim=1)
        assert (predicted == labels).float().mean().item() > 0.95

    def test_negative_ridge_rejected(self):
        with pytest.raises(ValueError):
            RidgeReadout(4, 2, ridge=-1.0)

    def test_none_ridge_rejected_with_a_readable_message(self):
        """None 是调用方的错——报出来的话要指得出是这个参数。"""
        with pytest.raises(ValueError, match="ridge 不能是 None"):
            RidgeReadout(4, 2, ridge=None)

    def test_solving_before_accumulating_is_an_error(self):
        with pytest.raises(ValueError):
            RidgeReadout(4, 2).solve()


class TestRidgeAccumulatorWiring:
    """回归：默认参数下造累积器曾经崩在 ``readout.py`` 的类型检查上。

    当时 ``train()`` 把 ``args.readout_ridge``（默认 ``None``）直接传给构造函数，
    于是「不给 λ 就在验证集上选」这条**默认路径**——也就是文档里写的、验收记录用的那条——
    一进来就 ``TypeError``。CI 的冒烟步骤抓到；本地的验收运行没有，因为它当时走的是另一条分支。
    """

    def test_default_cli_args_build_the_accumulator(self):
        args = build_parser().parse_args(["--smoke"])
        assert args.readout_ridge is None  # 默认：在验证集上选 λ
        accumulator = build_ridge_readout(args, torch.device("cpu"))
        assert accumulator.ridge > 0
        assert accumulator.xtx.shape == (args.width + 1, args.width + 1)

    def test_explicit_ridge_still_reaches_the_candidates(self):
        """显式给了 λ 时，构造函数不吃它、但选择那一步要吃到——这里钉住参数确实被解析。"""
        args = build_parser().parse_args(["--smoke", "--readout-ridge", "1e-3"])
        assert args.readout_ridge == pytest.approx(1e-3)
        build_ridge_readout(args, torch.device("cpu"))

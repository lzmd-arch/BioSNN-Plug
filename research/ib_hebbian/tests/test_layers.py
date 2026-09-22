"""``research.ib_hebbian.layers`` 的测试。

这一组测试的重心不是"能不能算"，而是**局部性到底成不成立**——那是计划书 §3.3 与 §9
（「无全局反向传播」）所依赖的性质，必须被断言，而不是靠文档里的一句话。
"""

from __future__ import annotations

import pytest
import torch

from research.ib_hebbian.layers import (
    DEFAULT_DIVNORM_POWER,
    LocalObjectiveLayer,
    grouped_activity,
)
from research.ib_hebbian.objective import centered_label_kernel


def make_label_kernel(m: int, n_classes: int = 4) -> torch.Tensor:
    return centered_label_kernel(torch.arange(m) % n_classes, n_classes)


class TestGroupedActivity:
    def test_shape_is_batch_by_groups(self):
        z = torch.randn(5, 12)
        assert grouped_activity(z, 3, exponent=0.8).shape == (5, 3)

    def test_matches_the_hand_computed_value(self):
        """Eq. (14)：u = (δ + Σ(∘z)²)/c，v = u^q。"""
        z = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        # 居中后 [-1.5,-0.5,0.5,1.5]，平方和 5，c=4，δ=0 → u = 1.25
        v = grouped_activity(z, 1, exponent=1.0, smoothing_delta=0.0, center=False)
        assert v.item() == pytest.approx(1.25)

    def test_centering_subtracts_the_group_mean(self):
        z = torch.randn(4, 8)
        uncentered = grouped_activity(z, 4, exponent=0.8, center=False)
        centered = grouped_activity(z, 4, exponent=0.8, center=True)
        torch.testing.assert_close(centered, uncentered - uncentered.mean(dim=-1, keepdim=True))

    def test_centered_output_sums_to_zero_across_groups(self):
        z = torch.randn(6, 16)
        v = grouped_activity(z, 4, exponent=0.8, center=True)
        torch.testing.assert_close(v.sum(dim=-1), torch.zeros(6), atol=1e-6, rtol=0)

    def test_exponent_changes_the_scale(self):
        z = torch.randn(3, 8) + 2.0
        low = grouped_activity(z, 2, exponent=0.2, center=False)
        high = grouped_activity(z, 2, exponent=0.8, center=False)
        assert not torch.allclose(low, high)

    def test_rejects_indivisible_width(self):
        with pytest.raises(ValueError, match="无法等分"):
            grouped_activity(torch.randn(2, 10), 3, exponent=0.8)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match=r"\(m, n_features\)"):
            grouped_activity(torch.randn(2, 3, 4), 2, exponent=0.8)


class TestLayerConstruction:
    def test_hidden_layers_have_no_bias(self):
        """论文 D.1：隐藏层没有偏置项，只有输出层有。"""
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        assert layer.linear.bias is None

    def test_grouping_exponent_is_one_minus_divnorm_power(self):
        """官方 argparse 明确要求 q = 1 − p；取错不会报错，只会安静掉点。"""
        layer = LocalObjectiveLayer(8, 4, n_groups=2, divnorm_power=0.2)
        assert layer.grouping_exponent == pytest.approx(0.8)

    def test_rejects_indivisible_width(self):
        with pytest.raises(ValueError, match="无法等分"):
            LocalObjectiveLayer(8, 6, n_groups=4)

    def test_optimizer_only_owns_this_layer(self):
        """每层一个优化器是「局部」的机械保证：没有共享优化器就没有跨层梯度累积。"""
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        owned = {id(p) for group in layer.optimizer.param_groups for p in group["params"]}
        assert owned == {id(p) for p in layer.parameters()}


class TestLocality:
    """这一组是模块存在的理由。"""

    def test_output_carries_no_gradient(self):
        """层输出必须已切断计算图——否则下游能顺着它回传到本层权重。"""
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        out = layer(torch.randn(5, 8))
        assert not out.requires_grad
        assert out.grad_fn is None

    def test_output_is_detached_even_without_an_update(self):
        """不更新时也切断。否则「梯度不跨层」这条就依赖"上一次有没有更新"这个脆弱前提。"""
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        assert layer(torch.randn(5, 8), update=False).grad_fn is None

    def test_rejects_an_input_that_carries_gradient(self):
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        x = torch.randn(5, 8, requires_grad=True)
        with pytest.raises(ValueError, match="局部性假设被破坏"):
            layer(x, make_label_kernel(5), update=True)

    def test_requires_a_label_kernel_when_updating(self):
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        with pytest.raises(ValueError, match="label_kernel"):
            layer(torch.randn(5, 8), update=True)

    def test_downstream_layer_leaves_upstream_weights_untouched(self):
        """核心断言：在第一层之上再叠一层，**不会**改变第一层的权重更新。

        这是「无全局反向传播」的可测形式。叠上第二层之后：
        * 第二层的输入是第一层的输出，而后者已被 detach，所以第二层的损失里没有
          通往第一层权重的路径；
        * 第二层用自己那个优化器，不会把梯度累积到第一层的参数上。
        因此第一层的权重在两种情形下必须逐位相同。
        """
        torch.manual_seed(0)
        x = torch.randn(12, 16)
        y_kernel = make_label_kernel(12)

        alone = LocalObjectiveLayer(16, 8, n_groups=2, learning_rate=0.5)
        with_downstream = LocalObjectiveLayer(16, 8, n_groups=2, learning_rate=0.5)
        downstream = LocalObjectiveLayer(8, 8, n_groups=2, learning_rate=0.5)
        with_downstream.load_state_dict(alone.state_dict())

        # 情形一：只有一层
        alone(x, y_kernel, update=True)

        # 情形二：同样的层，后面再接一层
        hidden = with_downstream(x, y_kernel, update=True)
        downstream_weights_before = downstream.linear.weight.detach().clone()
        downstream(hidden, y_kernel, update=True)

        torch.testing.assert_close(
            alone.linear.weight, with_downstream.linear.weight, rtol=0, atol=0
        )
        # 下游自己也确实更新了（证明这条测试不是"下游啥也没干"而平凡通过）
        assert not torch.equal(downstream.linear.weight, downstream_weights_before)

    def test_no_gradient_reaches_the_input(self):
        """输入侧必须干净：更新的梯度只落在本层权重上。"""
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        x = torch.randn(6, 8)
        layer(x, make_label_kernel(6), update=True)
        assert x.grad is None
        assert layer.linear.weight.grad is not None


class TestUpdate:
    def test_weights_change_after_an_update(self):
        layer = LocalObjectiveLayer(8, 4, n_groups=2, learning_rate=0.5)
        before = layer.linear.weight.detach().clone()
        layer(torch.randn(8, 8), make_label_kernel(8), update=True)
        assert not torch.equal(before, layer.linear.weight)

    def test_weights_are_untouched_without_an_update(self):
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        before = layer.linear.weight.detach().clone()
        layer(torch.randn(8, 8), update=False)
        torch.testing.assert_close(before, layer.linear.weight, rtol=0, atol=0)

    def test_repeated_updates_reduce_the_local_objective(self):
        """目标该下降——否则优化器与目标的接法就是错的。

        用同一个批反复更新（过拟合一个批），局部目标应当显著下降。这条能抓住
        "梯度符号反了""优化器没接上参数"这类错误，而那两类错误都不会报错。
        """
        torch.manual_seed(3)
        layer = LocalObjectiveLayer(16, 8, n_groups=2, learning_rate=0.05)
        x = torch.randn(16, 16)
        y_kernel = make_label_kernel(16)

        first = layer.measure_local_objective(x, y_kernel)
        for _ in range(40):
            layer(x, y_kernel, update=True)
        last = layer.measure_local_objective(x, y_kernel)

        assert last < first, f"局部目标没有下降：{first:.5f} → {last:.5f}"

    def test_measure_does_not_update(self):
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        before = layer.linear.weight.detach().clone()
        value = layer.measure_local_objective(torch.randn(6, 8), make_label_kernel(6))
        assert isinstance(value, float)
        torch.testing.assert_close(before, layer.linear.weight, rtol=0, atol=0)

    def test_divnorm_power_default_matches_the_paper(self):
        """论文 D.5：pHSIC/backprop 的除法归一化与分组用 p = 0.2、δ = 1。"""
        layer = LocalObjectiveLayer(8, 4, n_groups=2)
        assert layer.divnorm_power == pytest.approx(DEFAULT_DIVNORM_POWER)
        assert layer.smoothing_delta == pytest.approx(1.0)

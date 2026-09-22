"""``research.ib_hebbian.divisive_norm`` 的测试。

重点不是"算出来了"，而是**按论文 Eq. (13)/(17) 算出来的**——所以关键用例是与朴素
直算的逐元素对照，而不是对照另一份实现。
"""

from __future__ import annotations

import math

import pytest
import torch

from research.ib_hebbian.divisive_norm import (
    DivisiveNormalization,
    grouped_variance,
)


def naive_divisive_norm(z: torch.Tensor, n_groups: int, power: float, delta: float) -> torch.Tensor:
    """按 Eq. (13)/(17) 逐组朴素直算，作为对照基准。"""
    batch, n_features = z.shape
    group_size = n_features // n_groups
    out = torch.empty_like(z)
    for b in range(batch):
        for g in range(n_groups):
            lo, hi = g * group_size, (g + 1) * group_size
            block = z[b, lo:hi]
            centered = block - block.mean()
            variance = (delta + (centered**2).sum()) / group_size
            out[b, lo:hi] = centered / variance**power
    return out


class TestGroupedVariance:
    def test_matches_the_hand_computed_value(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # 1 批、1 组、4 个神经元
        # 居中后 = [-1.5, -0.5, 0.5, 1.5]，平方和 = 2.25+0.25+0.25+2.25 = 5
        expected = (0.0 + 5.0) / 4
        assert grouped_variance(x, n_groups=1, smoothing_delta=0.0).item() == pytest.approx(
            expected
        )

    def test_delta_is_divided_by_the_group_size(self):
        """Eq. (13) 写的是 δ/c，不是 δ。写错了 δ 就不是尺度无关的平滑了。"""
        x = torch.zeros(1, 1, 8)
        assert grouped_variance(x, 1, smoothing_delta=8.0).item() == pytest.approx(1.0)

    def test_zero_variance_is_rescued_by_delta(self):
        assert grouped_variance(torch.zeros(1, 1, 4), 1, smoothing_delta=1.0).item() > 0

    def test_constant_input_has_zero_centered_variance(self):
        x = torch.full((1, 1, 6), 3.5)
        assert grouped_variance(x, 1, 0.0).item() == pytest.approx(0.0)

    def test_groups_are_independent(self):
        """组内方差只由本组决定——这是"分组"的全部意义。"""
        x = torch.zeros(1, 2, 4)
        x[0, 1, :] = torch.tensor([1.0, -1.0, 1.0, -1.0])
        variance = grouped_variance(x, n_groups=2, smoothing_delta=0.0)
        assert variance[0, 0].item() == pytest.approx(0.0)
        assert variance[0, 1].item() == pytest.approx(1.0)


class TestDivisiveNormalization:
    def test_matches_the_naive_equation(self):
        torch.manual_seed(0)
        z = torch.randn(7, 12)
        module = DivisiveNormalization(n_groups=3, power=0.2, smoothing_delta=1.0)
        expected = naive_divisive_norm(z, n_groups=3, power=0.2, delta=1.0)
        torch.testing.assert_close(module(z), expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("power", [0.0, 0.2, 0.5, 1.0])
    def test_matches_the_naive_equation_across_powers(self, power):
        torch.manual_seed(1)
        z = torch.randn(5, 8) * 3.0
        module = DivisiveNormalization(n_groups=4, power=power, smoothing_delta=1.0)
        expected = naive_divisive_norm(z, n_groups=4, power=power, delta=1.0)
        torch.testing.assert_close(module(z), expected, rtol=1e-5, atol=1e-6)

    def test_output_is_centered_within_each_group(self):
        """∘z 是按组居中的，所以每组输出之和必须是 0。"""
        torch.manual_seed(2)
        z = torch.randn(6, 12)
        out = DivisiveNormalization(n_groups=3, power=0.2, smoothing_delta=1.0)(z)
        per_group_sums = out.reshape(6, 3, -1).sum(dim=-1)
        torch.testing.assert_close(per_group_sums, torch.zeros(6, 3), atol=1e-5, rtol=0)

    def test_p_half_matches_group_normalization_style_scaling(self):
        """p=0.5 时等价于按组标准差归一（论文：等价于除法归一化 / group norm）。

        对单个样本的单个组：居中后除以 sqrt(u)，其中 u = (δ + Σc²)/n。
        """
        z = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
        out = DivisiveNormalization(n_groups=1, power=0.5, smoothing_delta=0.0)(z)
        centered = z - z.mean()
        rms = math.sqrt((centered**2).sum().item() / 4)
        torch.testing.assert_close(out, centered / rms, rtol=1e-5, atol=1e-6)

    def test_shape_is_preserved(self):
        z = torch.randn(4, 16)
        assert DivisiveNormalization(4)(z).shape == z.shape

    def test_rejects_indivisible_feature_count(self):
        with pytest.raises(ValueError, match="无法等分"):
            DivisiveNormalization(n_groups=5)(torch.randn(2, 12))

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match=r"\(batch, n_features\)"):
            DivisiveNormalization(n_groups=2)(torch.randn(2, 3, 4))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_groups": 0}, "n_groups"),
            ({"n_groups": 2, "power": 1.5}, "power"),
            ({"n_groups": 2, "power": -0.1}, "power"),
            ({"n_groups": 2, "smoothing_delta": -1.0}, "smoothing_delta"),
        ],
    )
    def test_rejects_invalid_hyperparameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            DivisiveNormalization(**kwargs)

    def test_gradients_flow_to_the_input(self):
        """归一化要参与梯度——它是网络的一部分，不是只在损失里出现。"""
        z = torch.randn(3, 8, requires_grad=True)
        DivisiveNormalization(n_groups=2)(z).sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

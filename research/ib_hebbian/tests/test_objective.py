"""``research.ib_hebbian.objective`` 的测试。

关键用例一律与**论文的显式公式**对照（Eq. 26/34/35 各自写一份朴素直算），而不是
与另一份自己的实现对照——后者只能证明前后一致，证明不了与论文一致。
"""

from __future__ import annotations

import math

import pytest
import torch

from research.ib_hebbian.objective import (
    biased_hsic,
    centered_label_kernel,
    gaussian_kernel,
    kernelized_bottleneck_objective,
    phsic,
    squared_distances,
)


def naive_squared_distances(x: torch.Tensor) -> torch.Tensor:
    """逐对直算 ‖a_i − a_j‖²，不走展开式。"""
    m = x.shape[0]
    out = torch.zeros(m, m, dtype=x.dtype)
    for i in range(m):
        for j in range(m):
            out[i, j] = ((x[i] - x[j]) ** 2).sum()
    return out


def naive_gaussian_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Eq. (26)：k(a_i,a_j) = exp(−‖a_i − a_j‖²/(2σ²))。"""
    return torch.exp(-naive_squared_distances(x) / (2.0 * sigma**2))


class TestSquaredDistances:
    def test_matches_the_naive_computation(self):
        torch.manual_seed(0)
        x = torch.randn(6, 5)
        torch.testing.assert_close(
            squared_distances(x), naive_squared_distances(x), rtol=1e-5, atol=1e-6
        )

    def test_diagonal_is_zero(self):
        torch.manual_seed(1)
        d2 = squared_distances(torch.randn(8, 3))
        torch.testing.assert_close(torch.diagonal(d2), torch.zeros(8), atol=1e-6, rtol=0)

    def test_is_symmetric(self):
        torch.manual_seed(2)
        d2 = squared_distances(torch.randn(5, 4))
        torch.testing.assert_close(d2, d2.transpose(0, 1))

    def test_never_negative(self):
        """展开式在两点很近时会有浮点负值；不夹住就会在 sqrt 之前变成 NaN 源头。"""
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0 + 1e-7]])
        assert (squared_distances(x) >= 0).all()

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match=r"\(m, d\)"):
            squared_distances(torch.randn(2, 3, 4))

    def test_gradient_is_finite_at_coincident_points(self):
        """这正是没用 ``torch.pdist`` 的原因：pdist 在 i=j 处梯度是 0/0。"""
        x = torch.zeros(3, 4, requires_grad=True)
        squared_distances(x).sum().backward()
        assert torch.isfinite(x.grad).all()


class TestGaussianKernel:
    @pytest.mark.parametrize("sigma", [1.0, 5.0])
    def test_matches_the_naive_computation(self, sigma):
        torch.manual_seed(3)
        x = torch.randn(6, 4)
        torch.testing.assert_close(
            gaussian_kernel(x, sigma), naive_gaussian_kernel(x, sigma), rtol=1e-6, atol=1e-7
        )

    def test_diagonal_is_one(self):
        torch.manual_seed(4)
        k = gaussian_kernel(torch.randn(7, 3), sigma=2.0)
        torch.testing.assert_close(torch.diagonal(k), torch.ones(7), atol=1e-6, rtol=0)

    def test_is_bounded_by_one(self):
        torch.manual_seed(5)
        k = gaussian_kernel(torch.randn(9, 6) * 10, sigma=1.0)
        assert k.max().item() <= 1.0 + 1e-6
        assert k.min().item() >= 0.0

    def test_sigma_shrinks_the_kernel(self):
        """σ 越小，不同点之间的核值越接近 0（相似性越局部）。"""
        x = torch.tensor([[0.0], [3.0]])
        assert gaussian_kernel(x, 1.0)[0, 1] < gaussian_kernel(x, 10.0)[0, 1]

    def test_rejects_non_positive_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            gaussian_kernel(torch.randn(2, 2), sigma=0.0)


class TestCenteredLabelKernel:
    def test_same_class_is_one(self):
        labels = torch.tensor([0, 0, 3, 3])
        k = centered_label_kernel(labels, n_classes=10)
        assert k[0, 1].item() == pytest.approx(1.0)
        assert k[2, 3].item() == pytest.approx(1.0)

    def test_different_class_is_minus_one_over_n_minus_one(self):
        """Eq. (35)：异类为 −1/(n−1)，不是 −1。"""
        labels = torch.tensor([0, 5])
        k = centered_label_kernel(labels, n_classes=10)
        assert k[0, 1].item() == pytest.approx(-1.0 / 9.0)

    def test_two_classes_gives_minus_one(self):
        """n=2 时异类恰好是 −1——这时信号退化成 ±1 的纯符号。"""
        k = centered_label_kernel(torch.tensor([0, 1, 1]), n_classes=2)
        assert k[0, 1].item() == pytest.approx(-1.0)
        assert k[1, 2].item() == pytest.approx(1.0)

    def test_is_symmetric_with_unit_diagonal(self):
        labels = torch.tensor([0, 1, 2, 0])
        k = centered_label_kernel(labels, n_classes=3)
        torch.testing.assert_close(k, k.transpose(0, 1))
        torch.testing.assert_close(torch.diagonal(k), torch.ones(4), rtol=0, atol=1e-6)

    def test_signal_takes_only_two_values(self):
        """论文称它为"二值教学信号"——这是可断言的性质。"""
        labels = torch.tensor([0, 1, 2, 3, 4, 0])
        k = centered_label_kernel(labels, n_classes=5)
        assert len(set(k.flatten().tolist())) == 2

    def test_rejects_a_single_class(self):
        with pytest.raises(ValueError, match="n_classes"):
            centered_label_kernel(torch.tensor([0, 0]), n_classes=1)


class TestPhsicAndHsic:
    def test_phsic_matches_the_explicit_double_sum(self):
        """Eq. (34)：
        (1/m²)Σ_ij k^a_ij k^b_ij − (1/m²)Σ_ij k^a_ij · (1/m²)Σ_ql k^b_ql
        """
        torch.manual_seed(6)
        a = torch.rand(6, 6)
        b = torch.rand(6, 6)
        m = 6
        expected = (a * b).sum() / m**2 - (a.sum() / m**2) * (b.sum() / m**2)
        assert phsic(a, b).item() == pytest.approx(expected.item(), rel=1e-6)

    def test_phsic_is_zero_when_the_two_are_independent_of_each_other(self):
        """pHSIC 度量的是核之间的协变：把 B 换成常数，协变应为 0。"""
        torch.manual_seed(7)
        a = torch.rand(5, 5)
        constant = torch.full((5, 5), 0.7)
        assert phsic(a, constant).item() == pytest.approx(0.0, abs=1e-7)

    def test_biased_hsic_matches_the_centering_matrix_form(self):
        """论文注释称它等于 tr(HAHB)/m²，H = I − 11ᵀ/m。

        **要求 A、B 对称**——这是核矩阵的固有性质（k(a_i,a_j) = k(a_j,a_i)），
        也是那条等式成立的前提：推导中要把行均值与列均值合并成同一个量。
        拿非对称矩阵来测会失败，但那不是实现的问题，是输入不合法。
        """
        torch.manual_seed(8)
        # float64：测的是代数恒等式，不该被 float32 的精度噪声挡住
        raw_a = torch.rand(7, 7, dtype=torch.float64)
        raw_b = torch.rand(7, 7, dtype=torch.float64)
        a = (raw_a + raw_a.T) / 2
        b = (raw_b + raw_b.T) / 2
        m = 7
        h = torch.eye(m, dtype=torch.float64) - torch.ones(m, m, dtype=torch.float64) / m
        expected = (h @ a @ h * b).sum() / m**2
        assert biased_hsic(a, b).item() == pytest.approx(expected.item(), rel=1e-12)

    def test_biased_hsic_on_a_real_kernel_matrix(self):
        """真实用法就是高斯核矩阵，这条走的是不会误设前提的路径。"""
        torch.manual_seed(11)
        a = gaussian_kernel(torch.randn(9, 4, dtype=torch.float64), sigma=2.0)
        b = gaussian_kernel(torch.randn(9, 4, dtype=torch.float64), sigma=3.0)
        m = 9
        h = torch.eye(m, dtype=torch.float64) - torch.ones(m, m, dtype=torch.float64) / m
        expected = (h @ a @ h * b).sum() / m**2
        assert biased_hsic(a, b).item() == pytest.approx(expected.item(), rel=1e-10)

    def test_phsic_is_an_upper_bound_on_hsic_for_a_gaussian_kernel(self):
        """论文 Eq. (32)：pHSIC(A,A) − HSIC(A,A) = 2·Var(μ) ≥ 0。"""
        torch.manual_seed(9)
        k = gaussian_kernel(torch.randn(12, 4), sigma=3.0)
        assert phsic(k, k).item() >= biased_hsic(k, k).item() - 1e-7

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="形状必须一致"):
            phsic(torch.rand(3, 3), torch.rand(4, 4))


class TestKernelizedBottleneckObjective:
    def _batch(self, m: int = 8, d: int = 5, n_classes: int = 4, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(m, d, generator=generator)
        labels = torch.arange(m) % n_classes
        return z, centered_label_kernel(labels, n_classes)

    def test_matches_the_equation_assembled_by_hand(self):
        z, y_kernel = self._batch()
        expected = phsic(gaussian_kernel(z, 5.0), gaussian_kernel(z, 5.0)) - 2.0 * phsic(
            y_kernel, gaussian_kernel(z, 5.0)
        )
        got = kernelized_bottleneck_objective(z, y_kernel, sigma=5.0, gamma=2.0)
        assert got.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_gradient_flows_and_is_finite(self):
        z, y_kernel = self._batch()
        z = z.clone().requires_grad_(True)
        kernelized_bottleneck_objective(z, y_kernel).backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

    def test_gamma_zero_removes_the_teaching_signal(self):
        """γ=0 时目标只剩 pHSIC(Z,Z)，与标签完全无关——这保证标签确实只经 γ 起作用。"""
        z, y_kernel = self._batch()
        labels_swapped = centered_label_kernel(torch.tensor([1, 1, 2, 2, 3, 3, 0, 0]), n_classes=4)
        no_gamma = kernelized_bottleneck_objective(z, y_kernel, gamma=0.0)
        other_labels = kernelized_bottleneck_objective(z, labels_swapped, gamma=0.0)
        assert no_gamma.item() == pytest.approx(other_labels.item(), rel=1e-12)

    def test_gamma_changes_the_objective(self):
        z, y_kernel = self._batch()
        assert kernelized_bottleneck_objective(z, y_kernel, gamma=2.0).item() != pytest.approx(
            kernelized_bottleneck_objective(z, y_kernel, gamma=0.0).item()
        )

    def test_biased_mode_is_available_for_comparison(self):
        """论文对比过 HSIC 与 pHSIC；这条路径要能走通，哪怕不是主路径。"""
        z, y_kernel = self._batch()
        value = kernelized_bottleneck_objective(z, y_kernel, mode="biased")
        assert math.isfinite(value.item())

    def test_rejects_an_unknown_mode(self):
        z, y_kernel = self._batch()
        with pytest.raises(ValueError, match="mode"):
            kernelized_bottleneck_objective(z, y_kernel, mode="whatever")

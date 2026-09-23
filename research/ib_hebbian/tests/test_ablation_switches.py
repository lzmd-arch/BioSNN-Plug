"""消融开关与余弦核的测试。

这些开关是给「按论文 Table 1/表 3 的列做对照」用的，所以测试盯三件事：

1. **默认路径没被改动**（`grouping=True`/`divnorm=True`/`plausible`/`gaussian` 与加开关之前等价）；
2. **两个旁路真的旁路了**，且**不能**用参数糊过去——特别是两个已证实的陷阱：
   `n_groups=1`（目标恒 0、梯度恒 0、不报错）与 `divnorm_power=0`（那个模块在 power=0 时
   仍做组内居中、返回的不是 z）；
3. 开关从命令行一路透到 `kernelized_bottleneck_objective`。
"""

from __future__ import annotations

import pytest
import torch

from research.ib_hebbian.layers import LocalObjectiveLayer
from research.ib_hebbian.model import IBHebbianPerceptron
from research.ib_hebbian.objective import (
    cosine_kernel,
    gaussian_kernel,
    kernelized_bottleneck_objective,
)
from research.ib_hebbian.train_mnist import build_parser

DIM = 8


def _layer(**kwargs) -> LocalObjectiveLayer:
    return LocalObjectiveLayer(4, DIM, n_groups=2, **kwargs)


class TestCosineKernel:
    def test_matches_the_textbook_formula(self):
        """逐项与朴素定义比：``k_ij = <x_i,x_j> / (‖x_i‖‖x_j‖)``。"""
        torch.manual_seed(0)
        x = torch.randn(5, 3)
        kernel = cosine_kernel(x)
        for i in range(5):
            for j in range(5):
                expected = torch.dot(x[i], x[j]) / (x[i].norm() * x[j].norm())
                assert kernel[i, j].item() == pytest.approx(float(expected), abs=1e-6)

    def test_diagonal_is_one(self):
        torch.manual_seed(1)
        kernel = cosine_kernel(torch.randn(4, 6))
        assert torch.allclose(torch.diagonal(kernel), torch.ones(4), atol=1e-6)

    def test_zero_row_does_not_produce_nan(self):
        """全零行没有定义的方向——按官方写法只在范数为 0 时兜底，不能出 NaN。"""
        x = torch.zeros(3, 4)
        x[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        kernel = cosine_kernel(x)
        assert torch.isfinite(kernel).all()

    def test_gaussian_and_cosine_are_different_kernels(self):
        """两个核不是同一个东西（否则这一组对照就没意义了）。"""
        torch.manual_seed(2)
        x = torch.randn(6, 4)
        assert not torch.allclose(cosine_kernel(x), gaussian_kernel(x, sigma=5.0))


class TestKernelSwitchDispatch:
    def test_kernel_argument_selects_the_kernel(self):
        torch.manual_seed(3)
        z = torch.randn(6, 4)
        label_kernel = torch.eye(6)
        gaussian = kernelized_bottleneck_objective(z, label_kernel, kernel="gaussian")
        cosine = kernelized_bottleneck_objective(z, label_kernel, kernel="cossim")
        assert gaussian.item() != cosine.item()

    def test_unknown_kernel_is_rejected(self):
        with pytest.raises(ValueError, match="kernel"):
            kernelized_bottleneck_objective(torch.randn(3, 2), torch.eye(3), kernel="linear")


class TestGroupingSwitch:
    def test_grouping_off_feeds_the_raw_activity_to_the_kernel(self):
        """`grouping=False` 就是论文的 plain 变体：活动**直接进核**，不做分组、不做跨组居中。"""
        torch.manual_seed(4)
        layer = _layer(grouping=False)
        z = torch.randn(6, DIM)
        label_kernel = torch.eye(6)
        expected = kernelized_bottleneck_objective(
            z, label_kernel, sigma=layer.sigma, gamma=layer.gamma
        )
        assert layer.local_objective(z, label_kernel).item() == pytest.approx(
            expected.item(), rel=1e-6
        )

    def test_grouping_on_is_the_default_and_differs(self):
        torch.manual_seed(5)
        z = torch.randn(6, DIM)
        label_kernel = torch.eye(6)
        assert _layer().local_objective(z, label_kernel).item() != pytest.approx(
            _layer(grouping=False).local_objective(z, label_kernel).item()
        )

    def test_single_group_is_rejected_with_a_pointer_to_the_right_switch(self):
        """**静默失效的陷阱**：n_groups=1 且默认居中时目标恒为 0、梯度恒为 0，且不报错。

        所以这里显式拦掉，并在报错里指明「无分组」该用什么。
        """
        with pytest.raises(ValueError, match="grouping=False"):
            LocalObjectiveLayer(4, DIM, n_groups=1)

    def test_single_group_would_have_been_silently_degenerate(self):
        """把陷阱本身钉下来：单组 + 居中时 grouped_activity 恒为 0（这就是要拦它的理由）。

        这里直接查那个中间量，而不是构造一个本该被拒绝的层。
        """
        from research.ib_hebbian.layers import grouped_activity

        z = torch.randn(4, DIM)
        grouped = grouped_activity(z, 1, exponent=0.8, smoothing_delta=1.0, center=True)
        assert torch.count_nonzero(grouped).item() == 0


class TestDivnormSwitch:
    def test_divnorm_off_returns_the_activity_itself(self):
        """旁路整个模块：输出就是 ``nonlinearity(linear(x))``（detach 之后）。"""
        torch.manual_seed(6)
        layer = _layer(divnorm=False, dropout_p=0.0)
        x = torch.randn(5, 4)
        out = layer.forward(x)
        expected = layer.nonlinearity(layer.linear(x)).detach()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_divnorm_off_is_not_the_same_as_power_zero(self):
        """`--divnorm-power 0` **不是**「无除法归一化」：那个模块仍做组内居中。"""
        torch.manual_seed(7)
        x = torch.randn(5, 4)
        bypassed = _layer(divnorm=False, dropout_p=0.0).forward(x)
        zero_power = _layer(divnorm=True, divnorm_power=0.0, dropout_p=0.0).forward(x)
        assert not torch.allclose(bypassed, zero_power)

    def test_divnorm_on_is_the_default(self):
        """默认路径与加开关之前等价：输出 = dropout(divisive_norm(z))。

        注意要拿**同一个层**比——两个独立构造的层权重不同，比它们没有意义。
        """
        torch.manual_seed(8)
        layer = _layer(dropout_p=0.0)
        x = torch.randn(5, 4)
        assert layer.divnorm is True
        expected = layer.divisive_norm(layer.nonlinearity(layer.linear(x)).detach())
        assert torch.allclose(layer.forward(x), expected, atol=1e-6)


class TestCLIDefaultsAndPlumbing:
    def test_cli_defaults_are_the_acceptance_configuration(self):
        args = build_parser().parse_args([])
        assert args.hsic_estimate_mode == "plausible"
        assert args.kernel == "gaussian"
        assert args.grouping == "on"
        assert args.divnorm == "on"

    def test_switches_reach_the_layers(self):
        model = IBHebbianPerceptron(
            16,
            width=8,
            n_layers=2,
            n_classes=3,
            n_groups=2,
            objective_mode="biased",
            kernel="cossim",
            grouping=False,
            divnorm=False,
        )
        for layer in model.layers:
            assert layer.objective_mode == "biased"
            assert layer.kernel == "cossim"
            assert layer.grouping is False
            assert layer.divnorm is False

    def test_default_model_is_unchanged(self):
        model = IBHebbianPerceptron(16, width=8, n_layers=2, n_classes=3, n_groups=2)
        for layer in model.layers:
            assert (layer.objective_mode, layer.kernel, layer.grouping, layer.divnorm) == (
                "plausible",
                "gaussian",
                True,
                True,
            )

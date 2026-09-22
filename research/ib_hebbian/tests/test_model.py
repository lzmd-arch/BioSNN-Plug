"""``research.ib_hebbian.model`` 的测试。"""

from __future__ import annotations

import pytest
import torch

from research.ib_hebbian.model import IBHebbianPerceptron, preprocess_mnist


class TestPreprocessMnist:
    def test_maps_to_minus_one_to_one(self):
        """论文 D.4：(x/255 − 0.5)/0.5。0 → −1，255 → +1，127.5 → 0。"""
        images = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(3, 1, 1)
        out = preprocess_mnist(images)
        assert out[0, 0].item() == pytest.approx(-1.0)
        assert out[2, 0].item() == pytest.approx(1.0)

    def test_flattens_and_returns_float(self):
        images = torch.zeros(4, 28, 28, dtype=torch.uint8)
        out = preprocess_mnist(images)
        assert out.shape == (4, 28 * 28)
        assert out.dtype == torch.float32

    def test_midpoint_maps_to_zero(self):
        images = torch.full((1, 2, 2), 127.5)
        assert preprocess_mnist(images)[0, 0].item() == pytest.approx(0.0, abs=1e-6)


class TestArchitecture:
    def test_hidden_layers_have_no_bias_but_readout_does(self):
        """论文 D.1：隐藏层无偏置，输出层有。"""
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=2, n_groups=4)
        assert all(layer.linear.bias is None for layer in model.layers)
        assert model.readout.bias is not None

    def test_layer_count_and_output_shape(self):
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=3, n_groups=4, n_classes=10)
        assert len(model.layers) == 3
        logits = model(torch.randn(5, 28 * 28), torch.zeros(5, dtype=torch.long))
        assert logits.shape == (5, 10)

    def test_each_layer_has_its_own_optimizer(self):
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=3, n_groups=4)
        owned = [
            {id(p) for group in layer.optimizer.param_groups for p in group["params"]}
            for layer in model.layers
        ]
        assert len({frozenset(s) for s in owned}) == 3, "各层优化器不应共享参数"

    def test_readout_optimizer_owns_only_the_readout(self):
        """读出的交叉熵只该更新读出——否则交叉熵就成了跨层反向传播的入口。"""
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=2, n_groups=4)
        owned = {id(p) for group in model.readout_optimizer.param_groups for p in group["params"]}
        expected = {id(p) for p in model.readout.parameters()}
        assert owned == expected

    def test_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            IBHebbianPerceptron(28 * 28, width=32, n_layers=0, n_groups=4)

    def test_rejects_width_not_divisible_by_groups(self):
        with pytest.raises(ValueError, match="无法等分"):
            IBHebbianPerceptron(28 * 28, width=30, n_layers=2, n_groups=4)


class TestLocalityAtModelLevel:
    def test_readout_gradient_cannot_reach_hidden_layers(self):
        """核心断言：读出的交叉熵反向传播后，隐藏层权重**没有**梯度。

        隐藏层的输出在层内已被 detach，所以交叉熵的图里根本没有通往隐藏层权重的
        路径。这是"无全局反向传播"在模型层面的可测形式。
        """
        torch.manual_seed(0)
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=3, n_groups=4)
        x = torch.randn(8, 28 * 28)
        labels = torch.arange(8) % 10

        model.readout_optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(x, labels), labels).backward()

        assert model.readout.weight.grad is not None
        for index, layer in enumerate(model.layers):
            assert layer.linear.weight.grad is None, f"第 {index} 层收到了交叉熵的梯度"

    def test_hidden_updates_do_not_depend_on_the_readout(self):
        """换掉读出权重，隐藏层的更新必须逐位不变。"""
        torch.manual_seed(1)
        x = torch.randn(8, 28 * 28)
        labels = torch.arange(8) % 10

        model_a = IBHebbianPerceptron(28 * 28, width=32, n_layers=2, n_groups=4)
        model_b = IBHebbianPerceptron(28 * 28, width=32, n_layers=2, n_groups=4)
        model_b.load_state_dict(model_a.state_dict())
        with torch.no_grad():
            model_b.readout.weight.add_(5.0)

        model_a(x, labels, update_hidden=True)
        model_b(x, labels, update_hidden=True)

        for layer_a, layer_b in zip(model_a.layers, model_b.layers, strict=True):
            torch.testing.assert_close(layer_a.linear.weight, layer_b.linear.weight, rtol=0, atol=0)


class TestTraining:
    def test_evaluate_does_not_change_weights(self):
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=2, n_groups=4)
        before = [layer.linear.weight.detach().clone() for layer in model.layers]
        model.evaluate(torch.randn(6, 28 * 28), torch.arange(6) % 10)
        for layer, snapshot in zip(model.layers, before, strict=True):
            torch.testing.assert_close(layer.linear.weight, snapshot, rtol=0, atol=0)

    def test_repeated_steps_reduce_the_readout_loss(self):
        """同一个批上反复训练应当降低交叉熵——抓"优化器没接上""符号反了"这类错。"""
        torch.manual_seed(2)
        model = IBHebbianPerceptron(
            28 * 28, width=64, n_layers=2, n_groups=8, readout_learning_rate=0.05
        )
        x = torch.randn(16, 28 * 28)
        labels = torch.arange(16) % 4

        first = model.step(x, labels)
        for _ in range(30):
            last = model.step(x, labels)
        assert last < first, f"交叉熵没有下降：{first:.4f} → {last:.4f}"

    def test_step_requires_labels(self):
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=2, n_groups=4)
        with pytest.raises(ValueError, match="labels"):
            model(torch.randn(4, 28 * 28), None, update_hidden=True)

    def test_layer_objectives_returns_one_value_per_layer(self):
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=3, n_groups=4)
        values = model.layer_objectives(torch.randn(8, 28 * 28), torch.arange(8) % 10)
        assert len(values) == 3
        assert all(isinstance(v, float) for v in values)

    def test_layer_objectives_does_not_update(self):
        model = IBHebbianPerceptron(28 * 28, width=32, n_layers=2, n_groups=4)
        before = [layer.linear.weight.detach().clone() for layer in model.layers]
        model.layer_objectives(torch.randn(8, 28 * 28), torch.arange(8) % 10)
        for layer, snapshot in zip(model.layers, before, strict=True):
            torch.testing.assert_close(layer.linear.weight, snapshot, rtol=0, atol=0)

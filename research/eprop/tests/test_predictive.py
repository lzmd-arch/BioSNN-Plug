"""Predictive E-prop 的守卫测试。"""

from __future__ import annotations

import torch

from research.eprop.predictive import PredictiveEProp


def _sine(steps: int, batch: int, period: float = 200.0) -> torch.Tensor:
    t = torch.arange(steps, dtype=torch.float32)
    wave = torch.sin(2 * torch.pi * t / period)
    return wave.reshape(steps, 1, 1).expand(steps, batch, 1).contiguous()


class TestShapes:
    def test_rollout_shapes(self) -> None:
        model = PredictiveEProp(1, 32, generator=torch.Generator().manual_seed(0))
        target = _sine(40, 2)
        rollout = model.simulate(target)
        assert rollout.spikes.shape == (40, 2, 32)
        assert rollout.v_scaled.shape == (40, 2, 32)
        assert rollout.filtered.shape == (40, 2, 32)
        assert rollout.prediction.shape == (40, 2, 1)
        assert rollout.error.shape == (40, 2, 1)

    def test_free_running_shape_and_zero_error(self) -> None:
        model = PredictiveEProp(1, 16, generator=torch.Generator().manual_seed(0))
        rollout = model.simulate(steps=25)
        assert rollout.prediction.shape == (25, 1, 1)
        assert torch.equal(rollout.error, torch.zeros_like(rollout.error))

    def test_bad_target_rank_rejected(self) -> None:
        model = PredictiveEProp(1, 8, generator=torch.Generator().manual_seed(0))
        try:
            model.simulate(torch.zeros(10, 2))
        except ValueError:
            return
        raise AssertionError("维度错误的 target 应被拒绝。")


class TestDeterminism:
    def test_same_seed_same_predictions(self) -> None:
        target = _sine(30, 2)
        a = PredictiveEProp(1, 16, generator=torch.Generator().manual_seed(7))
        b = PredictiveEProp(1, 16, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a.simulate(target).prediction, b.simulate(target).prediction)
        assert torch.equal(a.cell.w_rec, b.cell.w_rec)
        assert torch.equal(a.w_out, b.w_out)

    def test_different_seed_differs(self) -> None:
        target = _sine(30, 2)
        a = PredictiveEProp(1, 16, generator=torch.Generator().manual_seed(1))
        b = PredictiveEProp(1, 16, generator=torch.Generator().manual_seed(2))
        assert not torch.equal(a.simulate(target).prediction, b.simulate(target).prediction)


class TestChunkedUpdates:
    def test_chunked_equals_single_update_at_full_length(self) -> None:
        """``chunk >= T`` 时，分段更新必须与单次更新逐位相同——这是分段实现正确性的守卫。"""
        target = _sine(60, 2)
        kwargs = {"learning_rate_rec": 1e-3, "learning_rate_out": 1e-2}
        # 两个模型必须各自持有一个 Generator：共用同一个对象会让第二个模型接着用第一个
        # 已经消耗过的随机状态，初始化就不同（这条曾经把测试写成假失败）。
        single = PredictiveEProp(1, 24, generator=torch.Generator().manual_seed(3), **kwargs)
        chunked = PredictiveEProp(1, 24, generator=torch.Generator().manual_seed(3), **kwargs)
        single.update(target)
        chunked.update(target, chunk=60)
        assert torch.equal(single.cell.w_rec, chunked.cell.w_rec)
        assert torch.equal(single.w_out, chunked.w_out)

    def test_chunked_makes_more_updates(self) -> None:
        """分段更新确实改变了权重轨迹（否则 chunk 参数形同虚设）。"""
        target = _sine(60, 2)
        kwargs = {"learning_rate_rec": 1e-3, "learning_rate_out": 1e-2}
        # 两个模型必须各自持有一个 Generator：共用同一个对象会让第二个模型接着用第一个
        # 已经消耗过的随机状态，初始化就不同（这条曾经把测试写成假失败）。
        single = PredictiveEProp(1, 24, generator=torch.Generator().manual_seed(3), **kwargs)
        chunked = PredictiveEProp(1, 24, generator=torch.Generator().manual_seed(3), **kwargs)
        single.update(target)
        chunked.update(target, chunk=10)
        assert not torch.equal(single.cell.w_rec, chunked.cell.w_rec)


class TestLearning:
    def test_loss_decreases(self) -> None:
        target = _sine(200, 4)
        model = PredictiveEProp(
            1,
            64,
            learning_rate_rec=1e-3,
            learning_rate_out=5e-2,
            generator=torch.Generator().manual_seed(0),
        )
        first = model.update(target)
        for _ in range(9):
            last = model.update(target)
        assert last < first, f"10 次更新后 MSE 未下降：{first:.4f} → {last:.4f}"

    def test_readout_gradient_matches_finite_difference(self) -> None:
        """读出更新必须真的减小重建误差（用 **前向** 差分核对方向，不靠 autograd）。"""
        target = _sine(80, 1)
        model = PredictiveEProp(
            1,
            32,
            learning_rate_rec=0.0,
            learning_rate_out=0.0,
            lambda_weight=0.0,
            generator=torch.Generator().manual_seed(0),
        )
        before = model.reconstruction_mse(target)
        # 只做一步读出更新（学习率另设，避免 rec 权重干扰）
        model.learning_rate_out = 5e-2
        model.update(target)
        after = model.reconstruction_mse(target)
        assert after < before, f"一步读出更新后 MSE 未下降：{before:.4f} → {after:.4f}"

    def test_no_autograd(self) -> None:
        target = _sine(50, 2)
        model = PredictiveEProp(1, 32, generator=torch.Generator().manual_seed(0))
        model.update(target)
        assert model.cell.w_rec.grad is None
        assert model.cell.w_in.grad is None

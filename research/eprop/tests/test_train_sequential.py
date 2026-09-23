"""``train_sequential`` 的两条不变量。

这个文件盯的不是「训练跑得对」（那由 `test_eprop.py` 承担），而是**两个会让结论静默失真的
性质**：

1. **评测不消耗随机流**——`--eval-every` 才敢被 runner 默认设成 1（每轮都评测）去画曲线。
   如果 `predict` 碰了随机数发生器，那么"把曲线画密一点"就会**改变学习结果**，而且不会报错：
   两个数字各自看着都正常，只是对不上。
2. **`--eval-every` 的默认值是 0**，即沿用原来的 ``epochs // 5`` 节奏——验收配置的行为
   必须逐字不变。
"""

from __future__ import annotations

import torch

from research.eprop.eprop import EPropLearner
from research.eprop.train_sequential import build_parser


def _model(**kwargs) -> EPropLearner:
    return EPropLearner(
        n_in=6,
        n_rec=8,
        n_out=4,
        generator=torch.Generator().manual_seed(1),
        **kwargs,
    )


class TestEvaluationDoesNotDisturbTraining:
    def test_predict_leaves_the_global_rng_state_untouched(self):
        """`predict` 全程只做前向，一个随机数都不该消耗。"""
        torch.manual_seed(0)
        model = _model()
        inputs = torch.rand(9, 3, 6)
        before = torch.get_rng_state().clone()
        model.predict(inputs)
        assert torch.equal(torch.get_rng_state(), before)

    def test_simulate_leaves_the_global_rng_state_untouched(self):
        torch.manual_seed(0)
        model = _model()
        inputs = torch.rand(9, 3, 6)
        before = torch.get_rng_state().clone()
        model.simulate(inputs)
        assert torch.equal(torch.get_rng_state(), before)

    def test_predict_is_deterministic(self):
        """同一个模型、同一份输入，跑两次必须逐位一致。"""
        torch.manual_seed(0)
        model = _model()
        inputs = torch.rand(9, 3, 6)
        assert torch.equal(model.predict(inputs), model.predict(inputs))

    def test_evaluating_midway_does_not_change_later_predictions(self):
        """「评测一次」不该影响**之后**的任何前向结果——这正是曲线密不密的区别所在。"""
        torch.manual_seed(0)
        model = _model()
        first = torch.rand(9, 3, 6)
        second = torch.rand(9, 3, 6)
        expected = model.predict(second)

        torch.manual_seed(0)
        model_again = _model()
        model_again.predict(first)  # 中间多插一次评测
        assert torch.equal(model_again.predict(second), expected)


class TestEvalEveryArgument:
    def test_default_is_zero_meaning_the_old_schedule(self):
        """0 = 沿用 `epochs // 5`，也就是验收配置的行为。"""
        assert build_parser().parse_args([]).eval_every == 0

    def test_parses_an_explicit_value(self):
        assert build_parser().parse_args(["--eval-every", "3"]).eval_every == 3

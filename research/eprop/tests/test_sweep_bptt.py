"""``research.eprop.sweep_bptt`` 的两条不变量。

这个文件盯的不是「BPTT 训得对不对」（那由 ``bptt_baseline.py`` 的既有测试承担），
而是**两个会让结论静默失真的性质**：

1. **配对是真的，不是一句"同 seed"。** 论文那句「同架构、同损失、同预算、同 seed，
   唯一差别是学习规则」全靠它。它成立的原因是 :func:`~research.common.seeding.derive_seed`
   按**用途名**派生（crc32，与调用顺序无关），而不是"记得传同一个种子"——后者一旦被
   谁调整了调用顺序就会静默失效，两个数字各自看着都正常，只是不可比。所以这里的基准
   取自 ``sweep_results/w2_long.log``：那次记录在案的运行（e-prop 0.8831）把它的种子行
   印在了复现记录块里。
2. **两个脉冲统计量与 e-prop 侧同定义。** BPTT 臂的活跃比例/稀疏度是照 ``EPropLearner``
   的同名方法重写的（BPTT 的前向不留脉冲），重写就会漂移。漂移了不会报错，只会让
   论文表格里那两列**跨臂不可比**。

训练本身不在测试范围内：要 GPU、要小时级，而且它跑的是 ``BPTTLearner``，那个由
``test_eprop.py`` 那类既有测试盯着。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from research.common.seeding import SeedBook
from research.eprop.bptt_baseline import BPTTLearner
from research.eprop.eprop import EPropLearner
from research.eprop.sweep_bptt import _spike_stats, apply_smoke, build_optimizer, build_parser

#: ``sweep_results/w2_long.log`` 里那次 e-prop 收敛运行（3 种子均值 0.8831，
#: ``--seeds 0-2 --epochs 250 --learning-rate-in 1e-2 --learning-rate-rec 1e-2``）
#: 复现记录块中的「随机种子」一行，逐字抄录。
RECORDED_W2_LONG_SEED_LINE = (
    "base=0；全局种子=1741413617；划分验证集=1435468909；"
    "权重初始化=3486733650；每批打乱=3990329895；脉冲编码=1681094898"
)

#: ``run_seed`` 里出现过的全部用途名（顺序无关，见下面的测试）。
PURPOSES = ("全局种子", "脉冲编码", "划分验证集", "权重初始化", "每批打乱")


def _seed_line(seed: int, purposes: tuple[str, ...]) -> str:
    """按给定顺序派生并渲染种子行。

    用 ``derive`` 而不是 ``apply``：这里只关心**派生出什么值**，而 ``apply`` 会顺带改写
    全局随机源，那会给别的测试留下跨用例的副作用。
    """
    book = SeedBook(base=seed)
    for purpose in purposes:
        book.derive(purpose)
    return book.render()


class TestPairingWithTheRecordedEpropRun:
    def test_seeds_match_the_w2_long_record(self):
        """BPTT 臂与那次 0.8831 的运行必须派生出同一组种子。"""
        assert _seed_line(0, PURPOSES) == RECORDED_W2_LONG_SEED_LINE

    def test_derivation_order_does_not_matter(self):
        """这正是配对能成立的原因：按用途名派生，与调用顺序无关。

        两个 book 用**相反**的顺序派生同一组用途名，结果必须一样。若哪天有人把
        ``derive_seed`` 换成顺序敏感的派生（例如"取计数器"），这条会红——而那时
        ``test_seeds_match_the_w2_long_record`` 也会红，两条一起说明配对断了。
        """
        forward = _seed_line(0, PURPOSES)
        backward = _seed_line(0, tuple(reversed(PURPOSES)))
        assert forward == backward

    def test_different_base_gives_different_seeds(self):
        """反向对照：换基准种子必须换出另一组值，否则上面两条可能只是"恒定不变"。"""
        assert _seed_line(1, PURPOSES) != RECORDED_W2_LONG_SEED_LINE


class TestSpikeStatsMatchTheEpropDefinition:
    """同一个生成器种子下，两种 learner 的循环层权重是**同一套**（都先建 ``ALIFCell``），
    所以这两个统计量可以直接对着比。"""

    @staticmethod
    def _pair():
        inputs = torch.rand(9, 3, 6, generator=torch.Generator().manual_seed(7))
        bptt = BPTTLearner(6, 8, 4, generator=torch.Generator().manual_seed(1))
        eprop = EPropLearner(6, 8, 4, generator=torch.Generator().manual_seed(1))
        return bptt, eprop, inputs

    def test_active_fraction_agrees(self):
        bptt, eprop, inputs = self._pair()
        active, _ = _spike_stats(bptt, inputs)
        assert active == pytest.approx(eprop.active_neuron_fraction(inputs))

    def test_sparsity_agrees(self):
        bptt, eprop, inputs = self._pair()
        _, sparsity = _spike_stats(bptt, inputs)
        assert sparsity == pytest.approx(eprop.spike_sparsity(inputs))

    def test_stats_are_fractions(self):
        bptt, _, inputs = self._pair()
        active, sparsity = _spike_stats(bptt, inputs)
        assert 0.0 <= active <= 1.0
        assert 0.0 <= sparsity <= 1.0


class TestOptimizerVariants:
    """变体 A/B 的差别**只许**是读出层那一个 lr。

    这里最要命的一条是参数集合：漏掉 ``w_out`` 不会报错，只会让它一直不更新，而准确率
    看上去仍然正常——正是那种"跑了六小时才发现读出层没动过"的错误。
    """

    @staticmethod
    def _model(learning_rate: float = 1e-3) -> BPTTLearner:
        return BPTTLearner(
            6, 8, 4, learning_rate=learning_rate, generator=torch.Generator().manual_seed(1)
        )

    @staticmethod
    def _optimized(model, optimizer) -> set[int]:
        return {id(p) for group in optimizer.param_groups for p in group["params"]}

    def test_variant_a_is_byte_for_byte_the_bpttlearner_recipe(self):
        """变体 A 必须与 ``BPTTLearner`` 自带的那个优化器同参数集合、同超参。

        基准取自 ``BPTTLearner.__init__`` 里**它自己造的那个**优化器（一份独立的构造），
        而不是拿 ``build_optimizer`` 的输出跟自己对——那样读两次同一个东西必然相等。
        """
        model = self._model(learning_rate=7e-3)
        ours = build_optimizer(model, variant="A", learning_rate=7e-3, learning_rate_out=5e-2)
        theirs = model.optimizer
        assert self._optimized(model, ours) == self._optimized(model, theirs)
        assert ours.param_groups[0]["lr"] == theirs.param_groups[0]["lr"] == 7e-3
        assert ours.param_groups[0]["momentum"] == theirs.param_groups[0]["momentum"] == 0.9

    def test_variant_b_optimizes_exactly_the_same_parameters(self):
        """两组加起来必须覆盖全部可训练参数——一个都不少、一个都不多。"""
        model = self._model()
        a = build_optimizer(model, variant="A", learning_rate=3e-3, learning_rate_out=5e-2)
        b = build_optimizer(model, variant="B", learning_rate=3e-3, learning_rate_out=5e-2)
        assert self._optimized(model, b) == self._optimized(model, a)
        assert {id(p) for p in model.parameters()} == self._optimized(model, b)

    def test_variant_b_splits_readout_from_hidden(self):
        """读出层拿到独立 lr，隐层拿扫描档；两组的 momentum 必须一样（唯一变量原则）。"""
        model = self._model()
        b = build_optimizer(model, variant="B", learning_rate=3e-3, learning_rate_out=5e-2)
        hidden, readout = b.param_groups
        assert hidden["lr"] == 3e-3
        assert readout["lr"] == 5e-2
        assert hidden["momentum"] == readout["momentum"] == 0.9
        assert [id(p) for p in readout["params"]] == [id(model.w_out)]
        assert self._optimized(model, b) - {id(model.w_out)} == {
            id(p) for p in model.cell.parameters()
        }

    def test_unknown_variant_is_an_error(self):
        """静默退回 A 会让调用方以为跑了 B——那是最坏的一种"成功"。"""
        with pytest.raises(ValueError, match="未知变体"):
            build_optimizer(self._model(), variant="C", learning_rate=1e-2, learning_rate_out=5e-2)

    def test_parser_defaults_to_variant_a(self):
        args = build_parser().parse_args([])
        assert args.variant == "A"
        assert args.learning_rate_out == 5e-2

    def test_parser_accepts_variant_b(self):
        assert build_parser().parse_args(["--variant", "B"]).variant == "B"


class TestSmokeScale:
    def test_smoke_shrinks_every_knob_that_costs_time(self):
        args = build_parser().parse_args(["--smoke"])
        apply_smoke(args)
        assert args.seeds == "0"
        assert args.epochs == 1
        assert args.n_rec == 32
        assert args.n_train == 256

    def test_smoke_writes_outside_the_recorded_results_dir(self):
        """冒烟产物不能落进 ``sweep_results/``——那是记录目录，塞一次无意义的运行进去
        会污染追踪（与 ``bptt_baseline`` 的冒烟同一条理由）。"""
        args = build_parser().parse_args(["--smoke"])
        apply_smoke(args)
        assert args.out_dir == Path("data/_smoke_bptt")

    def test_default_run_uses_the_whole_training_set(self):
        """``--n-train`` 默认 0 = 全量减验证集。**这是预算匹配的承重件**：它的默认值
        一旦变成某个有限数，BPTT 臂就不再与 e-prop 同预算，而数字看上去仍然正常。"""
        args = build_parser().parse_args([])
        assert args.n_train == 0

    def test_default_epochs_is_the_presweep_budget(self):
        assert build_parser().parse_args([]).epochs == 30

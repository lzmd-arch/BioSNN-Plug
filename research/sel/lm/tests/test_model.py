"""守卫：``TokenEProp`` 是**局部学习**实现——没有 autograd、状态会结转、同种子可复现。

这三条是"它算不算局部学习"的最低证据。缺了它们，后面的数字没法解释。
"""

from __future__ import annotations

import math

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, FAN_IN, LEN, BATCH = 40, 128, 16, 12, 4


def _model() -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=FAN_IN,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
    )


def test_no_tensor_requires_grad() -> None:
    """整个实现不许建计算图：任何一个张量带 requires_grad 就说明混进了 autograd。"""
    model = _model()
    for name in ("w_in", "w_out", "w_rec"):
        assert not getattr(model, name).requires_grad, name


def test_initial_ce_near_log_vocab() -> None:
    """未训练时 CE 应当接近 ln V；明显偏离说明抽头归一化或输入增益错了。"""
    model = _model()
    tokens = torch.randint(0, VOCAB, (BATCH, LEN))
    result = model.evaluate(tokens, model.new_trace(BATCH))
    assert abs(result["ce"] - math.log(VOCAB)) < 0.5, result


def test_state_carries_across_chunks() -> None:
    """带着状态跑第二块，结果必须与"重置后跑第二块"不同——否则结转没生效。"""
    model = _model()
    first = torch.randint(0, VOCAB, (BATCH, LEN))
    second = torch.randint(0, VOCAB, (BATCH, LEN))
    trace = model.new_trace(BATCH)
    model.run(first, trace, update=False)
    carried = model.evaluate(second, trace)["ce"]
    fresh = model.evaluate(second, model.new_trace(BATCH))["ce"]
    assert carried != fresh


def test_same_seed_same_result() -> None:
    """同种子两遍必须逐位一致（复现的基本要求）。"""
    tokens = torch.randint(0, VOCAB, (BATCH, LEN))
    a = _model().evaluate(tokens, _model().new_trace(BATCH))
    b = _model().evaluate(tokens, _model().new_trace(BATCH))
    assert a == b


def test_learning_changes_readout() -> None:
    """一次 update 之后读出权重必须变——否则"训练"是空的。"""
    model = _model()
    before = model.w_out.clone()
    tokens = torch.randint(0, VOCAB, (BATCH, LEN))
    model.run(tokens, model.new_trace(BATCH))
    assert not torch.equal(before, model.w_out)

"""守卫：A1 候选剪枝在 ``C = V`` 时必须与全量 softmax 逐位一致。

这条守卫是"候选剪枝是近似、不是换模型"的唯一证据。它一变红，A1 的结论就不能引用。
"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, BATCH, STEPS = 20, 16, 3, 12


def _model(**kwargs) -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        readout_taus=(2.0, 8.0),
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        **kwargs,
    )


def test_full_candidate_set_matches_dense_softmax() -> None:
    full = _model()
    all_cols = torch.arange(VOCAB).unsqueeze(0).expand(VOCAB, VOCAB).contiguous()
    cand = _model(candidates=all_cols)
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace_full, trace_cand = full.new_trace(BATCH), cand.new_trace(BATCH)
    for t in range(STEPS - 1):
        target = tokens[:, t + 1]
        loss_full, _ = full.step(tokens[:, t], target, trace_full)
        loss_cand, _ = cand.step(tokens[:, t], target, trace_cand)
        assert torch.allclose(loss_full, loss_cand, atol=1e-5)
    assert torch.allclose(full.w_out, cand.w_out, atol=1e-6)
    assert torch.equal(full.w_in, cand.w_in)


def test_narrow_candidates_only_touch_those_columns() -> None:
    """窄候选集只应改动候选列与 target 列——其它列必须逐位不变。"""
    cols = torch.arange(4).unsqueeze(0).expand(VOCAB, 4).contiguous()
    model = _model(candidates=cols)
    before = model.w_out.clone()
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace = model.new_trace(BATCH)
    for t in range(STEPS - 1):
        model.step(tokens[:, t], tokens[:, t + 1], trace)
    touched = set(cols.flatten().tolist()) | set(tokens.flatten().tolist())
    for column in range(VOCAB):
        if column in touched:
            continue
        assert torch.equal(model.w_out[:, column], before[:, column])

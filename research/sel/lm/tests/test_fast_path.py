"""守卫：全有效快路径必须与慢路径**逐位一致**（`active` 全 1 时两者代数上相等）。"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, BATCH, STEPS = 20, 16, 3, 10


def _make() -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        readout_taus=(2.0, 8.0),
        generator=torch.Generator().manual_seed(0),
        device="cpu",
    )


def test_all_valid_fast_path_is_bit_identical() -> None:
    fast, slow = _make(), _make()
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    tf, ts = fast.new_trace(BATCH), slow.new_trace(BATCH)
    for t in range(STEPS - 1):
        target = tokens[:, t + 1]  # 全有效：没有 -100
        loss_f, _ = fast.step(tokens[:, t], target, tf, all_valid=True)
        loss_s, _ = slow.step(tokens[:, t], target, ts, all_valid=False)
        assert torch.equal(loss_f, loss_s), t
        assert torch.equal(tf.v, ts.v) and torch.equal(tf.z, ts.z)
        assert torch.equal(tf.refractory, ts.refractory)
    assert torch.equal(fast.w_in, slow.w_in)
    assert torch.equal(fast.w_out, slow.w_out)

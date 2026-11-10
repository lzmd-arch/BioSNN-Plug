"""R12 守卫：BPTT 版的前向必须与 `TokenEProp` **逐位等价**。

这条不成立的话，`local vs BPTT` 的对照测的就不是同一个模型，
整个 2×2 也就没有意义。
"""

from __future__ import annotations

import torch

from research.sel.lm.bptt_lif import BPTTLIF
from research.sel.lm.model import TokenEProp

VOCAB, N_REC, STEPS = 37, 16, 24
TAUS = (2.0, 8.0, 32.0, 128.0)


def _pair():
    local = TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        recurrence="none",
        readout_taus=TAUS,
        pad_vocab=False,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
    )
    bptt = BPTTLIF(
        VOCAB,
        N_REC,
        readout_taus=TAUS,
        generator=torch.Generator().manual_seed(1),
        device="cpu",
    )
    with torch.no_grad():
        bptt.w_in.copy_(local.w_in)
        bptt.w_out.copy_(local.w_out)
    return local, bptt


def test_forward_matches_token_eprop() -> None:
    local, bptt = _pair()
    tokens = torch.randint(0, VOCAB, (3, STEPS), generator=torch.Generator().manual_seed(7))
    expected = local.run(tokens, local.new_trace(3), valid=None, update=False)["ce"]
    losses, _ = bptt(tokens)
    got = float(losses.mean())
    assert abs(got - expected) < 1e-5, f"BPTT 前向 {got} 与 TokenEProp {expected} 不等"


def test_forward_matches_with_padding_mask() -> None:
    local, bptt = _pair()
    tokens = torch.randint(0, VOCAB, (3, STEPS), generator=torch.Generator().manual_seed(8))
    valid = torch.ones(3, STEPS, dtype=torch.bool)
    valid[:, -6:] = False  # 尾部是 padding
    expected = local.run(tokens, local.new_trace(3), valid=valid, update=False)["ce"]
    losses, _ = bptt(tokens, valid)
    targets = torch.stack([tokens[:, t + 1] for t in range(STEPS - 1)], dim=1)
    masks = torch.stack([valid[:, t + 1] for t in range(STEPS - 1)], dim=1)
    keep = targets.masked_fill(~masks, -100) != -100
    got = float(losses[keep].mean())
    assert abs(got - expected) < 1e-5, f"带掩码时 {got} 与 {expected} 不等"


def test_surrogate_gradient_flows_to_input_weights() -> None:
    _, bptt = _pair()
    tokens = torch.randint(0, VOCAB, (2, STEPS), generator=torch.Generator().manual_seed(9))
    losses, _ = bptt(tokens)
    losses.mean().backward()
    assert bptt.w_in.grad is not None
    assert float(bptt.w_in.grad.abs().sum()) > 0, "伪导数没把梯度送回去"

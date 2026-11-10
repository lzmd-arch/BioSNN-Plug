"""S1 守卫：词表补零（V → 8 的倍数）必须**逐位不改变**输出与更新。"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, BATCH, STEPS = 21, 16, 3, 10  # 21 不是 8 的倍数，补到 24


def _make(**kwargs) -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        readout_taus=(2.0, 8.0),
        recurrence="none",  # 循环矩阵也吃 RNG 流；补零改变它在流里的位置，会污染对比
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        **kwargs,
    )


def test_padding_is_bit_exact() -> None:
    padded = _make()
    plain = _make(pad_vocab=False)
    assert padded.n_pad == 24 and padded.w_out.shape[1] == 24
    assert plain.n_pad == VOCAB
    # 补零版的前 VOCAB 列与不补零版同值 ⇒ 其余列只是从不被读到的填充
    plain.w_out.copy_(padded.w_out[:, :VOCAB])
    plain.w_in.copy_(padded.w_in)

    gen = torch.Generator().manual_seed(3)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace_a, trace_b = padded.new_trace(BATCH), plain.new_trace(BATCH)
    for t in range(STEPS - 1):
        target = tokens[:, t + 1]
        loss_a, _ = padded.step(tokens[:, t], target, trace_a)
        loss_b, _ = plain.step(tokens[:, t], target, trace_b)
        assert torch.equal(loss_a, loss_b), f"第 {t} 步的损失被补零改变"
    assert torch.equal(padded.w_out[:, :VOCAB], plain.w_out)


def test_padding_rounds_up_to_multiple_of_eight() -> None:
    for vocab in (1, 7, 8, 9, 3001):
        model = TokenEProp(
            vocab, 8, fan_in=4, generator=torch.Generator().manual_seed(0), device="cpu"
        )
        assert model.n_vocab == vocab
        assert model.n_pad % 8 == 0
        assert model.n_pad - vocab < 8

"""S5 守卫：分组归一化必须**真的接线**，而且 `norm_groups=1` 走历史路径。

这一条差点被漏掉：最初的冒烟配置里神经元根本不发放，`features` 恒零，
三种 `norm_groups` 给出逐位相同的损失——**看起来"通过"其实是没测到**。
所以守卫必须让网络发放，并且断言"分组确实改变了权重"。
"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC = 41, 16
TAUS = (2.0, 8.0, 32.0, 128.0)


def _make(**kwargs) -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        input_std=1.5,  # 必须够大，否则没有脉冲、features 恒零
        recurrence="none",
        readout_taus=TAUS,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        **kwargs,
    )


def _weights(norm_groups: int) -> tuple[torch.Tensor, float, float]:
    model = _make(norm_groups=norm_groups)
    tokens = torch.randint(0, VOCAB, (3, 9), generator=torch.Generator().manual_seed(1))
    result = model.run(tokens, model.new_trace(3), valid=None)
    return model.w_out.clone(), float(result["ce"]), float(result["firing_rate"])


def test_bounds_cover_the_readout_width() -> None:
    model = _make(norm_groups=4)
    assert model.norm_bounds[0][0] == 0
    assert model.norm_bounds[-1][1] == model.readout_width
    for (_, hi), (lo, _) in zip(model.norm_bounds, model.norm_bounds[1:], strict=False):
        assert hi == lo


def test_group_norm_actually_changes_the_update() -> None:
    base, _, fired = _weights(1)
    assert fired > 0, "配置没有发放，这一条测不到东西"
    for groups in (2, 4):
        other, _, _ = _weights(groups)
        assert not torch.equal(base, other), f"norm_groups={groups} 没有生效"


def test_group_norm_does_not_diverge() -> None:
    """分组归一化不能放大步长。

    少除一个 G 时，每步的总修正量是 Σ_g g = G·g，实测 CE 从 8.0 一路涨到 18.1（发散）。
    判据：分组后的 CE 与全局版同量级（比值 < 1.5）。
    """
    tokens = torch.randint(0, VOCAB, (3, 40), generator=torch.Generator().manual_seed(5))

    def ce(groups: int) -> float:
        model = _make(norm_groups=groups)
        return float(model.run(tokens, model.new_trace(3), valid=None)["ce"])

    base = ce(1)
    for groups in (2, 4):
        ratio = ce(groups) / base
        assert ratio < 1.5, f"norm_groups={groups} 的 CE 是全局版的 {ratio:.2f} 倍 ⇒ 步长被放大"


def test_group_norm_keeps_the_scale_reasonable() -> None:
    """分组归一化把每个组除自己的 norm，总更新量应与全局版同量级（不是 ×G）。"""
    base, _, _ = _weights(1)
    quarter, _, _ = _weights(4)
    ratio = float(quarter.abs().mean() / base.abs().mean())
    assert 0.3 < ratio < 3.0, ratio

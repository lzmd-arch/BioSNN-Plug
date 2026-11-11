"""守卫：训练基础设施的三处承重件——`learn_mask`、检查点轮转、评测的 rank cost。

这三样都不是"跑一次看看"能保证的：`learn_mask` 写反了会让探针悄悄退化成
"预测填充"；检查点没有轮转时，一次写坏等于把几十小时清零。
"""

from __future__ import annotations

import torch

from research.sel.lm.hierarchical import build_huffman_paths
from research.sel.lm.model import TokenEProp


def _model(vocab: int = 8, n_rec: int = 16, tree: bool = False) -> TokenEProp:
    return TokenEProp(
        vocab,
        n_rec,
        recurrence="none",
        readout_taus=(2.0,),
        tree=build_huffman_paths(torch.ones(vocab)) if tree else None,
        # 默认 input_std=0.12 时 5 步内神经元根本不发放，特征恒为 0，
        # 更新量也就恒为 0——那会让"掩码全 True 权重会变"这条守卫假绿。
        input_std=1.2,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
    )


def test_learn_mask_false_freezes_weights() -> None:
    """`learn_mask` 全 False ⇒ 一步之后权重逐位不变。"""
    model = _model()
    before = model.w_out.clone()
    trace = model.new_trace(2)
    mask = torch.zeros(2, dtype=torch.bool)
    for step in range(40):
        model.step(
            torch.tensor([step % 8, (step + 3) % 8]),
            torch.tensor([(step + 1) % 8, (step + 4) % 8]),
            trace,
            update=True,
            all_valid=True,
            learn_mask=mask,
        )
    assert torch.equal(model.w_out, before)


def test_learn_mask_true_updates_weights() -> None:
    """同样 5 步，`learn_mask` 全 True ⇒ 权重必须变（否则掩码根本没接上）。"""
    model = _model()
    before = model.w_out.clone()
    trace = model.new_trace(2)
    mask = torch.ones(2, dtype=torch.bool)
    for step in range(40):
        model.step(
            torch.tensor([step % 8, (step + 3) % 8]),
            torch.tensor([(step + 1) % 8, (step + 4) % 8]),
            trace,
            update=True,
            all_valid=True,
            learn_mask=mask,
        )
    assert not torch.equal(model.w_out, before)


def test_learn_mask_freezes_tree_readout() -> None:
    """树读出走的是同一套掩码，全 False 时树权重也必须不动。"""
    model = _model(tree=True)
    before = model.tree.weight.clone()
    trace = model.new_trace(2)
    mask = torch.zeros(2, dtype=torch.bool)
    for step in range(3):
        model.step(
            torch.tensor([step % 8, (step + 3) % 8]),
            torch.tensor([(step + 1) % 8, (step + 4) % 8]),
            trace,
            update=True,
            learn_mask=mask,
        )
    assert torch.equal(model.tree.weight, before)


def test_checkpoint_rotation(tmp_path) -> None:
    """第二次保存必须把上一份挪到 `.prev.pt`，主文件仍然可读。"""
    from research.sel.lm.run_1b import save_async, wait_saves

    path = tmp_path / "ckpt.pt"
    save_async({"step": 1, "tensors": {"a": torch.ones(4)}, "config": {}}, path)
    wait_saves()
    assert torch.load(path, weights_only=False)["step"] == 1
    save_async({"step": 2, "tensors": {"a": torch.zeros(4)}, "config": {}}, path)
    wait_saves()
    previous = path.with_suffix(".prev.pt")
    assert previous.exists(), "没有轮转，写坏一次就全丢"
    assert torch.load(previous, weights_only=False)["step"] == 1
    assert torch.load(path, weights_only=False)["step"] == 2

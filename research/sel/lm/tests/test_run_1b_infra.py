"""守卫：训练基础设施的三处承重件——`learn_mask`、检查点轮转、评测的 rank cost。

这三样都不是"跑一次看看"能保证的：`learn_mask` 写反了会让探针悄悄退化成
"预测填充"；检查点没有轮转时，一次写坏等于把几十小时清零。
"""

from __future__ import annotations

import pytest
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


def test_tree_node_ids_move_when_the_corpus_changes(tmp_path) -> None:
    """**V2 的那个坑**：同一棵词表、换一批频次，内部节点 id 会整体重排。

    这条不是在测某个函数，是在守住一个事实——所以它一旦变绿了，说明树构造改了，
    而"换语料续训"的整个论证要重新看。
    """
    a = build_huffman_paths(torch.tensor([10.0, 9.0, 8.0, 7.0, 1.0, 1.0, 1.0, 1.0]))
    b = build_huffman_paths(torch.tensor([1.0, 1.0, 1.0, 1.0, 9.0, 10.0, 8.0, 7.0]))
    same_nodes = torch.equal(a.nodes, b.nodes)
    same_bits = torch.equal(a.bits, b.bits)
    assert not (same_nodes and same_bits), "频次变了树却没变——这个前提要重写"


def test_saved_paths_round_trip_bit_exact(tmp_path) -> None:
    """冻结的树读回来必须**逐位**相同，否则"冻结"只是说法。"""
    from research.sel.lm.run_1b import load_paths, save_paths

    counts = torch.tensor([10.0, 9.0, 8.0, 7.0, 1.0, 1.0, 1.0, 1.0])
    paths = build_huffman_paths(counts)
    target = tmp_path / "tree.pt"
    save_paths(paths, target, vocab=len(counts), counts_key="deadbeef")

    back = load_paths(target)
    assert torch.equal(paths.nodes, back.nodes)
    assert torch.equal(paths.bits, back.bits)
    assert torch.equal(paths.length, back.length)
    assert paths.n_nodes == back.n_nodes
    assert paths.depth == back.depth


def test_counts_key_changes_with_the_corpus() -> None:
    """语料指纹要能把"换了语料"认出来。"""
    from research.sel.lm.run_1b import counts_key

    assert counts_key(torch.ones(8, dtype=torch.int64)) != counts_key(
        torch.arange(8, dtype=torch.int64)
    )


def test_corpus_counts_reports_a_cache_hit(tmp_path) -> None:
    """**续训能不能安全重建树，全靠这个布尔**：从缓存读 ⇒ 频次与上次相同 ⇒ 树逐位相同。

    它一旦恒为 True，换语料续训的守卫就整个失效；恒为 False 则会拦掉所有正常的续训。
    """
    import numpy as np

    from research.sel.lm.run_1b import corpus_counts

    corpus = tmp_path / "c.u16"
    np.array([1, 2, 3, 1, 2, 1], dtype=np.uint16).tofile(corpus)
    cache = tmp_path / "counts_4.npy"

    first, hit_first = corpus_counts([corpus], 4, cache)
    second, hit_second = corpus_counts([corpus], 4, cache)

    assert hit_first is False, "第一次没有缓存，必须现算"
    assert hit_second is True, "第二次必须走缓存"
    assert torch.equal(first, second)
    assert first.tolist() == [0, 3, 2, 1]


def test_effective_lr_in_halves_when_batch_doubles() -> None:
    """参考工作点固定：batch 翻倍，折算后的学习率必须减半。"""
    from research.sel.lm.run_1b import REF_BATCH, effective_lr_in

    assert effective_lr_in(1e-3, REF_BATCH, 1.0) == 1e-3
    assert effective_lr_in(1e-3, REF_BATCH * 2, 1.0) == 1e-3 / 2
    assert effective_lr_in(1e-3, REF_BATCH, 3.0) == 1e-3 / 3


def test_one_step_update_is_batch_invariant() -> None:
    """**这条量的是行为，不是公式。**

    训的是树读出，而树读出的信号**不除 batch**（稠密读出那条路径才除）。所以隐层更新
    `w_in += eps_inᵀ @ post` 会随 batch 线性增长；`effective_lr_in` 里的 `16/batch`
    正是补偿这一项。当年没补偿时，batch=1024 一步把发放率从 0.0087 打到 0.0006、
    CE 8.2 → 51.4。

    判据：**一步优化带来的权重变化，与 batch 无关**（用重复样本把 batch 放大 4×/16×）。
    """
    import torch

    from research.sel.lm.hierarchical import build_huffman_paths
    from research.sel.lm.run_1b import effective_lr_in

    def one_step(repeat: int, scale: float = 3.0) -> tuple[float, float]:
        batch = 2 * repeat
        paths = build_huffman_paths(torch.arange(8, 0, -1).float())
        model = TokenEProp(
            8,
            32,
            recurrence="none",
            readout_taus=(2.0,),
            input_std=1.2,
            tree=paths,
            tree_signal_scale=scale,
            lr_in=effective_lr_in(1e-3, batch, scale),
            generator=torch.Generator().manual_seed(0),
            device="cpu",
        )
        tokens = torch.tensor([1, 5]).repeat(repeat)
        targets = torch.tensor([2, 6]).repeat(repeat)
        trace = model.new_trace(tokens.numel())
        w_before = model.w_in.clone()
        tree_before = model.tree.weight.clone()
        model.step(tokens, targets, trace)
        return (
            float((model.w_in - w_before).abs().sum()),
            float((model.tree.weight - tree_before).abs().sum()),
        )

    base = one_step(1)
    assert base[0] > 0 and base[1] > 0, "更新量为 0，这条守卫等于没跑"
    for repeat in (4, 16):
        got = one_step(repeat)
        assert got[0] == pytest.approx(base[0], rel=1e-4), (repeat, got, base)
        assert got[1] == pytest.approx(base[1], rel=1e-4), (repeat, got, base)

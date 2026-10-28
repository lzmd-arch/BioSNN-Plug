"""事件驱动 e-prop 与 dense e-prop 的稀疏 mask 对照。"""

from __future__ import annotations

import torch

from research.eprop.eprop import EPropLearner
from research.eprop.event_driven import EventDrivenEProp


def test_matches_dense_eprop_on_the_same_mask():
    torch.manual_seed(0)
    n_in, n_rec, n_out, fan_in = 4, 7, 3, 2
    steps, batch = 6, 2
    generator = torch.Generator().manual_seed(1)
    inputs = (torch.rand(steps, batch, n_in, generator=generator) < 0.5).float()
    targets = torch.tensor([0, 2])

    sparse = EventDrivenEProp(
        n_in, n_rec, n_out, fan_in=fan_in, n_refractory=2, generator=torch.Generator().manual_seed(3)
    )
    dense = EPropLearner(
        n_in,
        n_rec,
        n_out,
        beta=0.0,
        threshold=sparse.threshold,
        n_refractory=2,
        generator=torch.Generator().manual_seed(3),
    )
    # 同一个 w_in / w_out；dense 的 w_rec 只在与 sparse 相同的 mask 上非零。
    with torch.no_grad():
        dense.cell.w_in.copy_(sparse.w_in)
        dense.w_out.copy_(sparse.w_out)
        dense.cell.w_rec.zero_()
        for post in range(n_rec):
            for slot in range(fan_in):
                dense.cell.w_rec[sparse.pre_index[post, slot], post] = sparse.w_rec[post, slot]

    w_in_before = sparse.w_in.detach().clone()
    w_out_before = sparse.w_out.detach().clone()
    w_rec_before = sparse.w_rec.detach().clone()
    dense_in_before = dense.cell.w_in.detach().clone()
    dense_out_before = dense.w_out.detach().clone()
    dense_rec_before = dense.cell.w_rec.detach().clone()

    sparse_loss = sparse.update(inputs, targets)
    dense_loss = dense.update(inputs, targets)
    assert sparse_loss == torch.tensor(dense_loss).item() or abs(sparse_loss - dense_loss) < 1e-6

    grad_in = sparse.w_in - w_in_before
    dense_grad_in = dense.cell.w_in - dense_in_before
    assert torch.allclose(grad_in, dense_grad_in, atol=1e-6, rtol=1e-5)

    grad_out = sparse.w_out - w_out_before
    dense_grad_out = dense.w_out - dense_out_before
    assert torch.allclose(grad_out, dense_grad_out, atol=1e-6, rtol=1e-5)

    for post in range(n_rec):
        for slot in range(fan_in):
            pre = sparse.pre_index[post, slot]
            assert torch.allclose(
                sparse.w_rec[post, slot] - w_rec_before[post, slot],
                dense.cell.w_rec[pre, post] - dense_rec_before[pre, post],
                atol=1e-6,
                rtol=1e-5,
            )

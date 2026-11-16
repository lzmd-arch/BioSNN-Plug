"""代谢内核：逐神经元能量 `E`，放电消耗、静默恢复、阈值随能量升高。

预注册：`research/sel/prereg-metabolic-core-v1.md`

## 为什么先做这个

目标路线里排第一，因为**没有可消耗、可恢复的内部资源，稳态、内源驱动、
自我模型都没有附着点**。这块地基的价值不在准确率，在
**"内部量是否自持"** —— 它由神经元自己的放电历史决定，不由输入决定。

## 一条机制，三档

    E  <-  clip( E - c*z + r*(1-z), 0, 1 )
    θ_eff = θ * (1 + k*(1 - E))

| 臂 | 参数 | 作用 |
| :--- | :--- | :--- |
| `ctrl` | `c=0, r=0`（`E ≡ 1`） | 无代谢对照 |
| `metab` | `c=0.05, r=0.00435, k=1` | 代谢开 |
| `starve` | `c=0.05, r=0` | 强制饥饿 |

## 判据（E-1 是核心）

| 编号 | 判据 | 线 |
| :--- | :--- | :--- |
| E-0 | 真实发放率 ∈ [0.03, 0.20] | 不过即无效 |
| **E-1** | 从**输入 token 序列**预测平均 `E` 的留出 R² | **< 0.5** |
| **E-2** | `corr(E, 发放率)` | **< −0.5** |
| E-3 | `metab` 准确率 ≥ `ctrl` − 5pp | 不塌 |
| E-4 | `starve` 发放率 < `metab` 的 0.5 倍 | 可预测地退化 |

**这一轮不追性能。** E-3 只要求不塌。
"""

from __future__ import annotations

# ruff: noqa: N806, N803
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import build, make_task
from scripts.instrument import calibrate, mlp_probe

OUT = _pathlib.Path("sweep_results/metabolic_core")
V, K = 64, 8
TAUS = (2.0, 16.0)

#: 能量参数。平衡点：c·f = r·(1−f)，f=0.08 ⇒ E 平衡在 0.5 附近
C_COST = 0.05
R_RECOVER = C_COST * 0.08 / 0.92
K_THETA = 1.0

ARMS = {
    "ctrl": (0.0, 0.0),  # E 恒为 1
    "metab": (C_COST, R_RECOVER),
    "starve": (C_COST, 0.0),
}


@torch.no_grad()
def run_metab(x, toks, dev, c: float, r: float, k: float = K_THETA, stride: int = 8):
    """返回 `(特征, 每步平均 E, 每步发放率, 总发放率)`。"""
    w_in, pre, wrec, alpha, _delay, _pool_id, _sw = x
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    E = torch.ones((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, e_trace, rate_trace = [], [], []
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        theta = 0.5 * (1.0 + k * (1.0 - E))
        nv = alpha * v + cur + rec - z * 0.5
        fresh = ((nv - theta) / theta > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        E = torch.clamp(E - c * nz + r * (1.0 - nz), 0.0, 1.0)
        v, z = nv, nz
        e_trace.append(E.mean(1))
        rate_trace.append(float(z.mean()))
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % stride == 0:
            feats.append(read.reshape(b, -1))
    return (
        torch.stack(feats, 1),
        torch.stack(e_trace, 1),
        torch.tensor(rate_trace, device=dev),
        sum(rate_trace) / T,
    )


def r2_from_tokens(e_trace: torch.Tensor, toks: torch.Tensor, L: int = 8, stride: int = 8) -> float:
    """从**输入 token 序列**预测平均 `E` 的留出 R²（按序列切分）。"""
    b, T = toks.shape
    pad = torch.zeros(b, L - 1, dtype=torch.long, device=toks.device)
    full = torch.cat([pad, toks], 1)
    idx = list(range(0, T, stride))
    X = torch.zeros(b, len(idx), L * V, device=toks.device)
    for i, t in enumerate(idx):
        win = full[:, t : t + L]
        for j in range(L):
            X[torch.arange(b, device=toks.device), i, j * V + win[:, j]] = 1.0
    X = X.reshape(-1, L * V)
    Y = e_trace[:, idx].reshape(-1)
    cut_seq = int(b * 0.8)
    cut = cut_seq * len(idx)
    Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lam = 1e-2 * Xtr.shape[0]
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=X.device)
    # 加截距列等价：先中心化目标，再补回均值
    ybar = Ytr.mean()
    W = torch.linalg.solve(A, Xtr.T @ (Ytr - ybar))
    pred = Xte @ W + ybar
    sse = float(((pred - Yte) ** 2).sum())
    sst = float(((Yte - Yte.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=384)
    ap.add_argument("--test-seq", type=int, default=96)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"n": args.n, "deg": args.deg, "c": C_COST, "r": R_RECOVER, "cells": {}}

    for arm, (c, r) in ARMS.items():
        tr0, _ = make_task(32, args.seq, K, 10, dev)
        w_in, pre, wrec, a_t, delay, pool_id = build(args.n, set(), 0, dev, args.deg)
        rest = (pre, wrec, a_t, delay, pool_id, set())
        ok_cal, sc, _pfr = calibrate(
            lambda s, _w=w_in, _r=rest, _t=tr0, _c=c, _rr=r: run_metab(
                (_w * s, *_r), _t, dev, _c, _rr
            )[3]
        )
        if not ok_cal:
            print(f"  {arm:6s} ✗ 标定失败，该臂无效", flush=True)
            continue
        accs, frs, r2s, cs = [], [], [], []
        for sd in seeds:
            tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
            te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
            toks = torch.cat([tr, te])
            st = torch.cat([trs, tes])[:, ::8]
            w2 = build(args.n, set(), sd, dev, args.deg)
            feat, e_tr, rate_tr, fr = run_metab((w2[0] * sc, *w2[1:], set()), toks, dev, c, r)
            m, _ = mlp_probe(feat, st, args.train_seq * st.shape[1], K)
            accs.append(m)
            frs.append(fr)
            r2s.append(r2_from_tokens(e_tr, toks))
            cs.append(corr(e_tr.mean(0), rate_tr))
            del feat, w2, toks, st
            torch.cuda.empty_cache()
        fr_mean = sum(frs) / len(frs)
        res["cells"][arm] = {
            "acc": sum(accs) / len(accs),
            "runs": accs,
            "firing": fr_mean,
            "r2_from_input": sum(r2s) / len(r2s),
            "corr_E_firing": sum(cs) / len(cs),
            "scale": sc,
        }
        print(
            f"  {arm:6s} scale {sc:5.2f} 发放 {fr_mean:.4f}  acc {sum(accs) / len(accs):.3f}  "
            f"R2(输入->E) {sum(r2s) / len(r2s):+.3f}  corr(E,发放) {sum(cs) / len(cs):+.3f}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )

    print("")
    print("判：")
    c = res["cells"]
    if "metab" not in c or "ctrl" not in c or "starve" not in c:
        print("  ✗ 有效臂不足，无法判")
        return 1
    m, ct, sv = c["metab"], c["ctrl"], c["starve"]
    print(
        f"  E-0 工作点：ctrl {ct['firing']:.4f} / metab {m['firing']:.4f} / starve {sv['firing']:.4f}"
    )
    print(
        f"  E-1 R2(输入->E) = {m['r2_from_input']:+.3f} < 0.5 ⇒ {'过' if m['r2_from_input'] < 0.5 else '不过'}"
    )
    print(
        f"  E-2 corr(E,发放) = {m['corr_E_firing']:+.3f} < -0.5 ⇒ {'过' if m['corr_E_firing'] < -0.5 else '不过'}"
    )
    print(
        f"  E-3 metab {m['acc']:.3f} ≥ ctrl {ct['acc']:.3f} − 0.05 = {ct['acc'] - 0.05:.3f} ⇒ "
        f"{'过' if m['acc'] >= ct['acc'] - 0.05 else '不过'}"
    )
    print(
        f"  E-4 starve 发放 {sv['firing']:.4f} < metab 的 0.5 倍 {m['firing'] * 0.5:.4f} ⇒ "
        f"{'过' if sv['firing'] < m['firing'] * 0.5 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

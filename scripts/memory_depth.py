"""时序记忆深度：线索保持任务（delay task）。

预注册：`research/sel/prereg-memory-depth-v1.md`

## 任务

`t = 0` 注入随机线索 `c ∈ [0, 16)`，之后跑与线索无关的背景流。
在特征步 `i`（`t = 8i`）上从网络状态解码 `c` —— **网络必须自己把 `c` 保持住**。

现有任务是循环的（`s_{t−k}` 由 `s_t` 唯一决定），用它测记忆深度是假的，
所以另造这一个。

## 为什么

v3/v4 里 `γ`（结构）与 `η`（活动）都通了机制、都没接到分类准确率上。
这一轮问：**是判据不敏感，还是机制本身没用？**
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
from scripts.development import (
    C_COST,
    D0,
    D1,
    R_RECOVER,
    homeostatic_rescale,
    prune,
)
from scripts.instrument import calibrate, mlp_probe

OUT = _pathlib.Path("sweep_results/memory_depth")
V, K = 64, 8
K_CUE = 16
TAUS = (2.0, 16.0)
K_THETA = 1.0
#: 线索持续注入的步数（单步太弱，扛不过背景流 —— 见 prereg amend2）
CUE_STEPS = 4

#: (beta, 代谢开, γ, η)  —— 四档取自 v3/v4
#: v6：只留结构变量（活动变量在 v5 已无可测效应）
ARMS = {
    "g0": (1.0, True, 0.0, 0.0),
    "g4": (1.0, True, 4.0, 0.0),
}
#: τ 的拟合窗（lag 1..5，信号在这段里还在地板之上）
TAU_LAGS = (1, 2, 3, 4, 5)


def make_delay_task(n_seq: int, T: int, seed: int, dev: str):
    """返回 `(背景 token 流, 线索)`。线索只在 t=0 出现一次。"""
    toks, _ = make_task(n_seq, T, K, seed + 1, dev)
    g = torch.Generator().manual_seed(seed * 7919 + 13)
    cue = torch.randint(0, K_CUE, (n_seq,), generator=g).to(dev)
    return toks, cue


@torch.no_grad()
def run_md(w_in, pre, wrec, toks, cue, dev, c, r, gamma, lam=0.95, alive=None, read_mask=None):
    """与 `development.run_dev` 同，但 `t = 0` 注入线索。返回 `(特征, 总发放率)`。"""
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    E = torch.ones((b, n), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, rate = [], 0.0
    for t in range(T):
        cur = w_in[cue] if t < CUE_STEPS else w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        theta = 0.5 * (1.0 + K_THETA * (1.0 - E))
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - theta) / theta > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        if alive is not None:
            nz = nz * alive  # 损伤：被切除的神经元不再放电
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        E = torch.clamp(E - c * nz + r * (1.0 - nz), 0.0, 1.0)
        v, z = nv, nz
        # **按存活神经元**求平均：死亡神经元恒为零，用 z.mean() 会把分母算大，
        # 于是"补偿到 0.08"实际把存活神经元驱动到 2 倍活动（v9 的混淆）
        # **分母必须含 batch 维**：z 是 (b, n)，alive 是 (n,)
        rate += float(z.sum() / (alive.sum() * b)) if alive is not None else float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if read_mask is not None:
            read = read * read_mask.view(1, 1, -1)
        if t % 8 == 0:
            feats.append(read.reshape(b, -1))
    return torch.stack(feats, 1), rate / T


def auc_of(curve, floor):
    """曲线下面积（lag>=1，减底板）。"""
    return sum(max(c - floor, 0.0) for c in curve[1:])


def tau_of(curve, floor, lags=TAU_LAGS):
    """指数拟合的衰减时间常数：log(acc-底板) ~ a - lag/tau。"""
    ys = [(lg, curve[lg] - floor) for lg in lags if lg < len(curve) and curve[lg] - floor > 1e-3]
    if len(ys) < 3:
        return float("nan")
    xs = torch.tensor([float(lg) for lg, _ in ys])
    ls = torch.tensor([__import__("math").log(y) for _, y in ys])
    xm, lm = xs.mean(), ls.mean()
    denom = float(((xs - xm) ** 2).sum())
    if denom <= 0:
        return float("nan")
    slope = float(((xs - xm) * (ls - lm)).sum()) / denom
    return -1.0 / slope if slope < 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--train-seq", type=int, default=384)
    ap.add_argument("--test-seq", type=int, default=96)
    ap.add_argument("--dev-seq", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    res: dict = {"n": args.n, "D0": D0, "D1": D1, "K_cue": K_CUE, "cells": {}}

    # 底板：多数类在测试集上的准确率
    majors = []
    for sd in seeds:
        _, cue = make_delay_task(args.train_seq + args.test_seq, args.seq, 10 + sd, dev)
        trp, tep = cue[: args.train_seq], cue[args.train_seq :]
        top = int(torch.bincount(trp, minlength=K_CUE).argmax())
        majors.append(float((tep == top).float().mean()))
    floor = sum(majors) / len(majors)
    res["majority"] = floor
    print(f"  底板(多数类) {floor:.3f}", flush=True)

    for arm, (beta, metab, gamma, eta) in ARMS.items():
        c, r = (C_COST, R_RECOVER) if metab else (0.0, 0.0)
        curves, frs = [], []
        for sd in seeds:
            cal_toks, cal_cue = make_delay_task(32, args.seq, 30 + sd, dev)
            w_in, pre, wrec, _a, _d, _p = build(args.n, set(), sd, dev, D0)

            def _rate(
                s, _w=w_in, _p=pre, _rw=wrec, _t=cal_toks, _cue=cal_cue, _cc=c, _rr=r, _g=gamma
            ):
                return run_md(_w * s, _p, _rw, _t, _cue, dev, _cc, _rr, _g)[1]

            ok_cal, sc, _pfr = calibrate(_rate)
            if not ok_cal:
                print(f"  {arm:5s} seed {sd} ✗ 标定失败", flush=True)
                continue
            dev_toks, _dev_cue = make_delay_task(args.dev_seq, args.seq, 30 + sd, dev)
            gen = torch.Generator().manual_seed(1000 + sd)
            # 发育期的痕迹：用 `development.run_dev` 的等价逻辑（线索注入不影响痕迹口径）
            from scripts.development import run_dev

            trace, _, e_tr, rate_before = run_dev(w_in * sc, pre, wrec, dev_toks, dev, c, r, gamma)
            e_bar = float(e_tr.mean())
            target = rate_before * (e_bar**eta)
            p_new, w_new = prune(pre, wrec, trace, D1, beta, 0.5, gen)
            w_new = homeostatic_rescale(w_in * sc, p_new, w_new, dev_toks, dev, c, r, target, gamma)
            tr, cue_tr = make_delay_task(args.train_seq, args.seq, 10 + sd, dev)
            te, cue_te = make_delay_task(args.test_seq, args.seq, 20 + sd, dev)
            toks = torch.cat([tr, te])
            cue = torch.cat([cue_tr, cue_te])
            feat, fr = run_md(w_in * sc, p_new, w_new, toks, cue, dev, c, r, gamma)
            n_steps = feat.shape[1]
            curve = []
            for i in range(n_steps):
                if args.train_seq < 64:
                    curve.append(float("nan"))
                    continue
                acc, _ = mlp_probe(feat[:, i], cue, args.train_seq, K_CUE)
                curve.append(acc)
            curves.append(curve)
            frs.append(fr)
            del feat, toks, cue, w_in, pre, wrec
            torch.cuda.empty_cache()
        if not curves:
            continue
        n_s = len(curves[0])
        mean_curve = [
            sum(cv[i] for cv in curves if cv[i] == cv[i])
            / max(1, sum(1 for cv in curves if cv[i] == cv[i]))
            for i in range(n_s)
        ]
        # 深度 = 最大的 lag 使 acc >= 底板 + 10pp
        thresh = floor + 0.10
        depth = -1
        for i, a in enumerate(mean_curve):
            if a >= thresh:
                depth = i
        res["cells"][arm] = {
            "curve": mean_curve,
            "curves": curves,
            "depth": depth,
            "auc": [auc_of(cv, floor) for cv in curves],
            "tau": [tau_of(cv, floor) for cv in curves],
            "firing": sum(frs) / len(frs),
            "gamma": gamma,
            "eta": eta,
        }
        print(
            f"  {arm:5s} γ={gamma:.0f} η={eta:.0f}  发放 {sum(frs) / len(frs):.4f}  "
            f"曲线 {[round(a, 3) for a in mean_curve]}  深度 {depth}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    if not cc:
        print("  ✗ 无有效臂")
        return 1
    thresh = floor + 0.10
    peaks = {k: max(v["curve"]) for k, v in cc.items()}
    md0 = all(pk >= thresh for pk in peaks.values())
    print(
        f"  MD-0 峰值每臂 ≥ {thresh:.3f}：{[round(x, 3) for x in peaks.values()]} "
        f"=> {'过' if md0 else '不过 => 作废'}"
    )
    if not md0:
        return 0
    md1 = all(
        all(v["curve"][i] >= v["curve"][i + 1] - 0.02 for i in range(len(v["curve"]) - 1))
        for v in cc.values()
    )
    print(f"  MD-1 曲线随 lag 单调下降（允许一步持平）=> {'过' if md1 else '不过'}")
    depths = {k: v["depth"] for k, v in cc.items()}
    print(f"  MD-2 深度：{depths}")

    pa = cc["g0"]["auc"]
    pb = cc["g4"]["auc"]
    da = [b - a for a, b in zip(pa, pb, strict=False)]
    pos = sum(1 for d in da if d > 0)
    n = len(da)
    print(f"  MD-4 AUC：逐种子 g4-g0 = {[round(d, 3) for d in da]}")
    print(f"        同号 {max(pos, n - pos)}/{n} => {'过' if max(pos, n - pos) == n else '不过'}")
    ta = cc["g0"]["tau"]
    tb = cc["g4"]["tau"]
    dt = [b - a for a, b in zip(ta, tb, strict=False)]
    ok_t = all(d == d for d in dt)
    pos_t = sum(1 for d in dt if d > 0)
    print(f"  MD-5 tau：逐种子 g4-g0 = {[round(d, 4) for d in dt]}")
    print(
        f"        同号 {max(pos_t, n - pos_t)}/{n} => "
        f"{'过' if ok_t and max(pos_t, n - pos_t) == n else '不过'}"
    )
    ra = (sum(pb) / len(pb)) / (sum(pa) / len(pa)) - 1.0
    print(f"  效应量：AUC 相对差 {ra * 100:+.1f}%（不作判据）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

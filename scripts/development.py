"""发育：过度生长 → 活动依赖修剪 → 稳态缩放；能量以 `E^γ` 进痕迹。

预注册：`research/sel/prereg-development-v3.md`

## 机制

```
s      <-  λ·s + mean_b( z_post · z_pre · E_post^γ )
score  =  s^β · (1 + 噪声)     # 每神经元保留 top-k
w_new  <-  稳态缩放到**修剪前的发放率**（突触缩放保的是活动，不是权重和）
```

- `γ = 0`：痕迹不带能量（参照）；`γ` 越大，痕迹越被能量主导；
- `β = 1`：修剪按痕迹；`β = 0`：随机修剪。

## 判据（v3）

| 编号 | 线 |
| :--- | :--- |
| C-0 | 窗口天花板 ≥ 底板 + 20pp；每臂 > 底板；发放在 [0.03, 0.20] |
| C-1 | `Jaccard(gγ, g0)` 随 γ **单调下降**（耦合确实在起作用） |
| C-2 | 最好的 γ 的 acc ≥ g0 + 3pp |
| C-3 | 结构两两 Jaccard < 0.8，且 acc 极差 < 2pp（渠化） |
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import build, make_task
from scripts.instrument import calibrate, mlp_probe
from scripts.structure_ladder import window_features

OUT = _pathlib.Path("sweep_results/development")
V, K = 64, 8
TAUS = (2.0, 16.0)
C_COST = 0.05
R_RECOVER = C_COST * 0.08 / 0.92
K_THETA = 1.0
D0, D1 = 32, 8

#: (beta, 代谢开, E 的指数 gamma, 设定点指数 eta)
ARMS = {
    "eta0": (1.0, True, 1.0, 0.0),
    "eta1": (1.0, True, 1.0, 1.0),
    "eta2": (1.0, True, 1.0, 2.0),
}


@torch.no_grad()
def run_dev(w_in, pre, wrec, toks, dev, c: float, r: float, gamma: float = 0.0, lam: float = 0.95):
    """返回 `(痕迹, 特征, 每步平均 E, 总发放率)`。`gamma=0` 时痕迹不带能量。"""
    b, T = toks.shape
    n, D = pre.shape
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    E = torch.ones((b, n), device=dev)
    trace = torch.zeros((n, D), device=dev)
    decays = torch.tensor([1.0 / t for t in TAUS], device=dev)
    read = torch.zeros((b, len(TAUS), n), device=dev)
    feats, e_trace, rate = [], [], 0.0
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for j in range(D):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        theta = 0.5 * (1.0 + K_THETA * (1.0 - E))
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - theta) / theta > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        E = torch.clamp(E - c * nz + r * (1.0 - nz), 0.0, 1.0)
        co = nz.unsqueeze(2) * nz[:, pre]
        if gamma > 0:
            co = co * E.unsqueeze(2).pow(gamma)
        trace = lam * trace + co.mean(0)
        v, z = nv, nz
        e_trace.append(E.mean(1))
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % 8 == 0:
            feats.append(read.reshape(b, -1))
    return trace, torch.stack(feats, 1), torch.stack(e_trace, 1), rate / T


@torch.no_grad()
def prune(pre, wrec, trace, k: int, beta: float, noise: float, gen):
    """按 `trace^beta` 偏置随机保留 top-k。"""
    n, D = pre.shape
    rnd = torch.rand(n, D, generator=gen).to(trace.device)
    score = trace.clamp(min=0).pow(beta) * (1.0 + noise * rnd)
    idx = score.argsort(1, descending=True)[:, :k]
    return pre.gather(1, idx), wrec.gather(1, idx)


@torch.no_grad()
def homeostatic_rescale(w_in, pre, wrec, toks, dev, c, r, target, gamma=0.0, tol=0.05, iters=12):
    """**按发放率**做稳态缩放（突触缩放保的是活动，不是权重和）。"""
    lo, hi = 0.05, 4.0
    w_best = wrec
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        w_try = wrec * mid
        _, _, _, fr = run_dev(w_in, pre, w_try, toks, dev, c, r, gamma)
        w_best = w_try
        if abs(fr - target) / max(target, 1e-9) < tol:
            return w_best
        if fr < target:
            lo = mid
        else:
            hi = mid
    return w_best


def jaccard(pa: torch.Tensor, pb: torch.Tensor, n: int) -> float:
    rows = torch.arange(n, device=pa.device).unsqueeze(1)
    ka = (rows * n + pa).reshape(-1)
    kb = (rows * n + pb).reshape(-1)
    inter = float(torch.isin(ka, kb).sum())
    union = float(ka.numel() + kb.numel() - inter)
    return inter / max(union, 1.0)


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
    res: dict = {"n": args.n, "D0": D0, "D1": D1, "cells": {}}

    # 底板与天花板（不经过网络）
    majors, ceils = [], []
    for sd in seeds:
        tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
        te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
        toks = torch.cat([tr, te])
        st = torch.cat([trs, tes])[:, ::8]
        trp, tep = st[: args.train_seq], st[args.train_seq :]
        top = int(torch.bincount(trp.reshape(-1), minlength=K).argmax())
        majors.append(float((tep == top).float().mean()))
        wf = window_features(toks, L=8, stride=8)
        ceils.append(mlp_probe(wf, st, args.train_seq * st.shape[1], K)[0])
        del toks, st, wf
    floor = sum(majors) / len(majors)
    ceiling = sum(ceils) / len(ceils)
    res["majority"] = floor
    res["ceiling"] = ceiling
    print(f"  底板(多数类) {floor:.3f}   窗口天花板 {ceiling:.3f}", flush=True)

    structures: dict = {}
    for arm, (beta, metab, gamma, eta) in ARMS.items():
        c, r = (C_COST, R_RECOVER) if metab else (0.0, 0.0)
        accs, frs, arms_struct = [], [], []
        for sd in seeds:
            tr0, _ = make_task(32, args.seq, K, 10, dev)
            w_in, pre, wrec, _a, _d, _p = build(args.n, set(), sd, dev, D0)
            ok_cal, sc, _pfr = calibrate(
                lambda s, _w=w_in, _p=pre, _rw=wrec, _t=tr0, _c=c, _r=r, _g=gamma: run_dev(
                    _w * s, _p, _rw, _t, dev, _c, _r, _g
                )[3]
            )
            if not ok_cal:
                print(f"  {arm:4s} seed {sd} ✗ 标定失败", flush=True)
                continue
            dev_toks, _ = make_task(args.dev_seq, args.seq, K, 30 + sd, dev)
            gen = torch.Generator().manual_seed(1000 + sd)
            trace, _, e_tr, rate_before = run_dev(w_in * sc, pre, wrec, dev_toks, dev, c, r, gamma)
            e_bar = float(e_tr.mean())
            # **设定点由代谢史决定**：更穷的个体把目标活动调得更低
            target = rate_before * (e_bar**eta)
            p_new, w_new = prune(pre, wrec, trace, D1, beta, 0.5, gen)
            w_new = homeostatic_rescale(w_in * sc, p_new, w_new, dev_toks, dev, c, r, target, gamma)
            tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
            te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
            toks = torch.cat([tr, te])
            st = torch.cat([trs, tes])[:, ::8]
            _, feat, _, fr = run_dev(w_in * sc, p_new, w_new, toks, dev, c, r, gamma)
            m, _ = mlp_probe(feat, st, args.train_seq * st.shape[1], K)
            accs.append(m)
            frs.append(fr)
            arms_struct.append(p_new.clone())
            del feat, toks, st, w_in, pre, wrec
            torch.cuda.empty_cache()
        if not accs:
            continue
        res["cells"][arm] = {
            "gamma": gamma,
            "eta": eta,
            "e_bar": e_bar,
            "target_rate": target,
            "acc": sum(accs) / len(accs),
            "runs": accs,
            "firing": sum(frs) / len(frs),
        }
        structures[arm] = arms_struct
        print(
            f"  {arm:4s} gamma={gamma:3.1f}  发放 {sum(frs) / len(frs):.4f}  "
            f"acc {sum(accs) / len(accs):.3f}  ({'/'.join(f'{a:.3f}' for a in accs)})",
            flush=True,
        )

    # 跨臂结构相似度（同种子配对）
    if structures.get("g0"):
        for arm in ("eta1", "eta2"):
            if structures.get(arm):
                nn = min(len(structures["g0"]), len(structures[arm]))
                js = [jaccard(structures[arm][i], structures["g0"][i], args.n) for i in range(nn)]
                res["cells"][arm]["jaccard_vs_g0"] = js
                print(f"  Jaccard({arm}, g0) = {[round(x, 3) for x in js]}", flush=True)
    for arm in ARMS:
        if structures.get(arm) and len(structures[arm]) >= 2:
            ss = structures[arm]
            js = [
                jaccard(ss[i], ss[j], args.n) for i in range(len(ss)) for j in range(i + 1, len(ss))
            ]
            res["cells"][arm]["jaccard_self"] = js

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    if "eta0" not in cc:
        print("  ✗ 参照臂 eta0 无效")
        return 1
    d01 = ceiling >= floor + 0.20
    d02 = all(v["acc"] > floor for v in cc.values())
    d03 = all(0.03 <= v["firing"] <= 0.20 for v in cc.values())
    print(
        f"  E-0 天花板 {ceiling:.3f} / 每臂>底板 / 工作点 => "
        f"{'过' if (d01 and d02 and d03) else '不过 => 作废'}"
    )
    if not (d01 and d02 and d03):
        return 0

    fr = [cc[a]["firing"] for a in ("eta0", "eta1", "eta2") if a in cc]
    mono = len(fr) == 3 and all(fr[i] > fr[i + 1] for i in range(2))
    print(
        f"  E-1 策略改变：发放 {[round(x, 4) for x in fr]} 随 eta 单调下降 => {'过' if mono else '不过'}"
    )

    base = cc["eta0"]["acc"]
    cand = [a for a in ("eta1", "eta2") if a in cc]
    best = max(cand, key=lambda a: cc[a]["acc"])
    gain = cc[best]["acc"] - base
    print(
        f"  E-2 能力：最好 {best} {cc[best]['acc']:.3f} - eta0 {base:.3f} = {gain * 100:+.1f}pp "
        f"(要 >=+3) => {'过' if gain >= 0.03 else '不过'}"
    )

    def _eff(a):
        return (cc[a]["acc"] - floor) / max(cc[a]["firing"], 1e-9)

    e0 = _eff("eta0")
    eb = max(cand, key=_eff)
    ratio = _eff(eb) / max(e0, 1e-12)
    print(
        f"  E-3 每脉冲能力：(acc-底板)/发放  eta0 {e0:.2f} / 最好 {eb} {_eff(eb):.2f} "
        f"= {ratio:.2f}x (要 >=1.15) => {'过' if ratio >= 1.15 else '不过'}"
    )
    return 0

    js = [max(cc[a].get("jaccard_vs_g0", [1.0])) for a in ("g1", "g2", "g4") if a in cc]
    mono = len(js) == 3 and all(js[i] > js[i + 1] for i in range(len(js) - 1))
    print(
        f"  C-1 剂量-反应：Jaccard(g1,g0)..(g4,g0) = {[round(x, 3) for x in js]} 单调下降 "
        f"=> {'过' if mono else '不过'}"
    )

    base = cc["g0"]["acc"]
    cand = [a for a in ("g1", "g2", "g4") if a in cc]
    best = max(cand, key=lambda a: cc[a]["acc"])
    gain = cc[best]["acc"] - base
    print(
        f"  C-2 能力：最好 {best} {cc[best]['acc']:.3f} - g0 {base:.3f} = {gain * 100:+.1f}pp "
        f"(要 >=+3) => {'过' if gain >= 0.03 else '不过'}"
    )

    a = cc[best]
    runs = a["runs"]
    jmax = max(a.get("jaccard_self", [1.0]))
    spread = max(runs) - min(runs)
    print(
        f"  C-3 渠化：{best} 结构两两 Jaccard 最大 {jmax:.3f} (要 <0.8)、"
        f"acc 极差 {spread * 100:.1f}pp (要 <2pp) => "
        f"{'过' if jmax < 0.8 and spread < 0.02 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

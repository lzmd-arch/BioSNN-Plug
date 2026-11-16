"""发育：过度生长 → 活动依赖修剪 → 稳态缩放。

预注册：`research/sel/prereg-development-v1.md`

## 机制

1. 起始入度 `D0 = 32`，目标 `D1 = 8`；
2. 跑一遍流，累积**共活动痕迹**（局部，无全局误差）：
   `s <- lam*s + mean_b( z_post · z_pre )`
3. 修剪：`score = s^β · (1 + 噪声)`，每个神经元保留 top-k；
   `β = 1` 活动驱动，`β = 0` 随机（对照）；
4. **稳态缩放**：存活突触重新归一化到修剪前的总驱动 ——
   否则"修剪了"与"驱动变小了"会混在一起。

## 判据

| 编号 | 线 |
| :--- | :--- |
| D-0 | 发育后发放在窗内，且 `acid` acc ≥ 底板 + 15pp |
| D-1 | 3 种子两两 Jaccard < 0.8，且 acc 极差 < 2pp（渠化） |
| D-2 | `acid` acc ≥ `rand` acc + 3pp |
| D-3 | `acid` 与 `acid_nometab` 结构 Jaccard < 0.9 |

**D-1 是定义性判据**：它同时排除"结构是随机的"（能力会散）与
"我们在搜索结构"（结构会收敛到同一个）。
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

OUT = _pathlib.Path("sweep_results/development")
V, K = 64, 8
TAUS = (2.0, 16.0)
C_COST = 0.05
R_RECOVER = C_COST * 0.08 / 0.92
K_THETA = 1.0
D0, D1 = 32, 8


@torch.no_grad()
def run_dev(w_in, pre, wrec, toks, dev, c: float, r: float, lam: float = 0.95):
    """跑一遍流，返回 `(痕迹, 特征, 每步平均 E, 总发放率)`。"""
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
        # 共活动痕迹：后突触放电 × 前突触放电（局部可得，无全局误差）
        trace = lam * trace + (nz.unsqueeze(2) * nz[:, pre]).mean(0)
        v, z = nv, nz
        e_trace.append(E.mean(1))
        rate += float(z.mean())
        read = read * decays.view(1, -1, 1) + z.unsqueeze(1)
        if t % 8 == 0:
            feats.append(read.reshape(b, -1))
    return trace, torch.stack(feats, 1), torch.stack(e_trace, 1), rate / T


@torch.no_grad()
def prune(pre, wrec, trace, k: int, beta: float, noise: float, gen):
    """按 `trace^beta` 偏置随机保留 top-k，并做稳态缩放。"""
    n, D = pre.shape
    score = trace.clamp(min=0).pow(beta) * (
        1.0 + noise * torch.rand(n, D, generator=gen).to(trace.device)
    )
    idx = score.argsort(1, descending=True)[:, :k]
    new_pre = pre.gather(1, idx)
    new_w = wrec.gather(1, idx)
    old_sum = wrec.sum(1, keepdim=True)
    new_w = new_w * (old_sum / new_w.sum(1, keepdim=True).clamp(min=1e-9))
    return new_pre, new_w


def jaccard(pa: torch.Tensor, pb: torch.Tensor, n: int) -> float:
    rows = torch.arange(n, device=pa.device).unsqueeze(1)
    ka = (rows * n + pa).reshape(-1)
    kb = (rows * n + pb).reshape(-1)
    inter = float(torch.isin(ka, kb).sum())
    union = float(ka.numel() + kb.numel() - inter)
    return inter / max(union, 1.0)


ARMS = {
    "acid": (1.0, True),
    "rand": (0.0, True),
    "acid_nometab": (1.0, False),
    "noprune": (1.0, True),
}


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
    majors = []
    for sd in seeds:
        tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
        te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
        st = torch.cat([trs, tes])[:, ::8]
        trp, tep = st[: args.train_seq], st[args.train_seq :]
        top = int(torch.bincount(trp.reshape(-1), minlength=K).argmax())
        majors.append(float((tep == top).float().mean()))
    floor = sum(majors) / len(majors)
    res["majority"] = floor
    print(f"  底板(多数类) {floor:.3f}", flush=True)

    all_structures: dict = {}
    for arm, (beta, metab) in ARMS.items():
        c, r = (C_COST, R_RECOVER) if metab else (0.0, 0.0)
        accs, frs, structures = [], [], []
        for sd in seeds:
            tr0, _ = make_task(32, args.seq, K, 10, dev)
            w_in, pre, wrec, _a, _d, _p = build(args.n, set(), sd, dev, D0)
            ok_cal, sc, _pfr = calibrate(
                lambda s, _w=w_in, _p=pre, _r=wrec, _t=tr0, _c=c, _rr=r: run_dev(
                    _w * s, _p, _r, _t, dev, _c, _rr
                )[3]
            )
            if not ok_cal:
                print(f"  {arm:12s} seed {sd} ✗ 标定失败", flush=True)
                accs.append(float("nan"))
                continue
            # 发育：在一段自己的活动上累积痕迹
            dev_toks, _ = make_task(args.dev_seq, args.seq, K, 30 + sd, dev)
            gen = torch.Generator().manual_seed(1000 + sd)
            trace, _, _, _ = run_dev(w_in * sc, pre, wrec, dev_toks, dev, c, r)
            if arm == "noprune":
                p_new, w_new = pre, wrec
            else:
                p_new, w_new = prune(pre, wrec, trace, D1, beta, 0.5, gen)
            # 评测
            tr, trs = make_task(args.train_seq, args.seq, K, 10 + sd, dev)
            te, tes = make_task(args.test_seq, args.seq, K, 20 + sd, dev)
            toks = torch.cat([tr, te])
            st = torch.cat([trs, tes])[:, ::8]
            feat, _, _, fr = run_dev(w_in * sc, p_new, w_new, toks, dev, c, r)
            m, _ = mlp_probe(feat, st, args.train_seq * st.shape[1], K)
            accs.append(m)
            frs.append(fr)
            structures.append(p_new.clone())
            del feat, toks, st
            torch.cuda.empty_cache()
        valid = [a for a in accs if a == a]
        fr_mean = sum(frs) / max(len(frs), 1)
        res["cells"][arm] = {
            "acc": sum(valid) / max(len(valid), 1),
            "runs": accs,
            "firing": fr_mean,
        }
        all_structures[arm] = structures
        if len(structures) >= 2:
            js = [
                jaccard(structures[i], structures[j], args.n)
                for i in range(len(structures))
                for j in range(i + 1, len(structures))
            ]
            res["cells"][arm]["jaccard"] = js
        print(
            f"  {arm:12s} 发放 {fr_mean:.4f}  acc {sum(valid) / max(len(valid), 1):.3f}  "
            f"({'/'.join(f'{a:.3f}' for a in accs)})",
            flush=True,
        )

    # 跨臂结构相似度（D-3）
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )

    print("")
    print("判：")
    cc = res["cells"]
    if "acid" not in cc:
        print("  ✗ 主臂无效")
        return 1
    a = cc["acid"]
    ok0 = 0.03 <= a["firing"] <= 0.20 and a["acc"] >= floor + 0.15
    print(f"  D-0 工作点 + 真解码：{'过' if ok0 else '不过 ⇒ 作废'}")
    if not ok0:
        return 1
    runs = [x for x in a["runs"] if x == x]
    jmax = max(a.get("jaccard", [1.0]))
    spread = max(runs) - min(runs)
    print(
        f"  D-1 渠化：结构两两 Jaccard 最大 {jmax:.3f}（要 <0.8）、acc 极差 {spread * 100:.1f}pp"
        f"（要 <2pp）⇒ {'过' if jmax < 0.8 and spread < 0.02 else '不过'}"
    )
    if "rand" in cc:
        g = a["acc"] - cc["rand"]["acc"]
        print(
            f"  D-2 活动依赖做功：acid − rand = {g * 100:+.1f}pp（要 ≥+3）⇒ {'过' if g >= 0.03 else '不过'}"
        )
    if "acid_nometab" in all_structures and all_structures.get("acid_nometab"):
        pairs = [
            jaccard(all_structures["acid"][i], all_structures["acid_nometab"][i], args.n)
            for i in range(min(len(all_structures["acid"]), len(all_structures["acid_nometab"])))
        ]
        res["cross_jaccard_acid_vs_nometab"] = pairs
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )
        j3 = max(pairs) if pairs else 1.0
        print(
            f"  D-3 E 塑造结构：acid vs acid_nometab 逐种子 Jaccard {[round(x, 3) for x in pairs]}"
            f" 最大 {j3:.3f}（要 <0.9）⇒ {'过' if j3 < 0.9 else '不过'}"
        )
    else:
        print("  D-3 缺少 acid_nometab 的结构，无法判")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

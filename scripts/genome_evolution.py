"""基因组与演化：用 7 个基因替代我手拍的 9 个量。

预注册：`research/sel/prereg-genome-evolution-v1.md`

## 硬纪律

**适应度不含准确率。** 生物选的是存活：可行性 + 抗损伤 + 稳定性 + 能量效率。
准确率只作报告量。

## 基因组（7 个实数，全部编码在单位超立方里）

`c` 代谢成本 · `r` 恢复率 · `k` 阈值调制 · `λ` 扩展半径 · `deg` 入度 ·
`A` 折叠幅度 · `target` 目标发放
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.instrument import FIRING_WINDOW, calibrate

OUT = _pathlib.Path("sweep_results/genome_evolution")
V = 64
L = 32
N = L * L
R = 8

GENES = ["c", "r", "k", "lam", "deg", "A", "target"]

#: 每个基因的 (低, 高, 是否对数)
BOUNDS = {
    "c": (0.005, 0.2, True),
    "r": (5e-4, 0.05, True),
    "k": (0.0, 3.0, False),
    "lam": (1.0, 16.0, True),
    "deg": (8.0, 64.0, True),
    "A": (0.0, 24.0, False),
    "target": (0.03, 0.20, True),
}

#: 我原来手拍的那一套
HAND = {"c": 0.05, "r": 0.00435, "k": 1.0, "lam": 4.0, "deg": 32.0, "A": 16.0, "target": 0.08}


def decode(u):
    """单位超立方 -> 实际基因值。"""
    g = {}
    for i, name in enumerate(GENES):
        lo, hi, log = BOUNDS[name]
        x = float(min(max(u[i], 0.0), 1.0))
        g[name] = (
            math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
            if log
            else lo + x * (hi - lo)
        )
    return g


def encode(g):
    """实际值 -> 单位超立方。"""
    u = []
    for name in GENES:
        lo, hi, log = BOUNDS[name]
        v = min(max(g[name], lo), hi)
        u.append(
            (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
            if log
            else (v - lo) / (hi - lo)
        )
    return u


@torch.no_grad()
def build_from_genes(g, seed, dev):
    """按 (λ, deg, A) 在一块折叠的环面薄片上建接线。"""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.arange(N)
    x = (idx % L).float().to(dev)
    y = (idx // L).float().to(dev)
    z = g["A"] * torch.sin(2.0 * math.pi * x / L)
    ax = torch.arange(-R, R + 1).float()
    ox, oy = torch.meshgrid(ax, ax, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev), oy.reshape(-1).to(dev)
    C = ox.numel()
    cx = (x.unsqueeze(1) + ox.unsqueeze(0)) % L
    cy = (y.unsqueeze(1) + oy.unsqueeze(0)) % L
    zc = g["A"] * torch.sin(2.0 * math.pi * cx / L)
    d2 = (ox.unsqueeze(0) ** 2 + oy.unsqueeze(0) ** 2).expand(N, C)
    d = torch.sqrt(d2 + (z.unsqueeze(1) - zc) ** 2)
    w = torch.exp(-d / g["lam"])
    w[:, (ox == 0) & (oy == 0)] = 0.0
    score = w * (1.0 + 0.5 * torch.rand(N, C, generator=gen).to(dev))
    k = round(g["deg"])
    idx_top = score.topk(k, dim=1).indices
    lin = (cy * L + cx).long()
    pre = lin.gather(1, idx_top)
    w_in = (torch.randn(V, N, generator=gen) / N**0.5).to(dev)
    wrec = torch.rand(N, k, generator=gen).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    return w_in, pre, wrec


@torch.no_grad()
def run_g(w_in, pre, wrec, toks, dev, c, r, k, alive=None):
    """代谢 + 逐神经元能量。返回 `(每步平均发放, 总发放)`。"""
    b, T = toks.shape
    n = w_in.shape[1]
    v = torch.zeros((b, n), device=dev)
    z = torch.zeros((b, n), device=dev)
    ref = torch.zeros((b, n), device=dev)
    E = torch.ones((b, n), device=dev)
    rates = []
    for t in range(T):
        cur = w_in[toks[:, t]]
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        theta = 0.5 * (1.0 + k * (1.0 - E))
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - theta) / theta > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        if alive is not None:
            nz = nz * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        E = torch.clamp(E - c * nz + r * (1.0 - nz), 0.0, 1.0)
        v, z = nv, nz
        rates.append(float(z.sum() / (alive.sum() * b)) if alive is not None else float(z.mean()))
    return torch.tensor(rates, device=dev), sum(rates) / T


def evaluate(g, seed, dev, toks):
    """适应度：可行性 3 + 抗损伤 4 + 稳定性 1 + 效率 1。**不含准确率。**"""
    w_in, pre, wrec = build_from_genes(g, seed, dev)

    def _rate(s, _a=None):
        return run_g(w_in * s, pre, wrec, toks, dev, g["c"], g["r"], g["k"], _a)[1]

    ok, sc, _fr = calibrate(_rate, target=g["target"])
    if not ok:
        return None
    rt, mean_f = run_g(w_in * sc, pre, wrec, toks, dev, g["c"], g["r"], g["k"])
    lo, hi = FIRING_WINDOW
    gen = torch.Generator().manual_seed(9000 + seed)
    alive = (torch.rand(N, generator=gen) >= 0.5).float().to(dev)
    _, mean_l = run_g(w_in * sc, pre, wrec, toks, dev, g["c"], g["r"], g["k"], alive)

    v = 1.0 if lo <= mean_f <= hi else 0.0
    d = 1.0 if lo <= mean_l <= hi else 0.0
    cv = float(rt.std() / max(rt.mean(), 1e-9))
    stab = math.exp(-cv)
    eff = max(0.0, min(1.0, 1.0 - (mean_f - lo) / (hi - lo)))
    return {
        "fitness": 3 * v + 4 * d + stab + eff,
        "viable": v,
        "damage_ok": d,
        "stability": stab,
        "efficiency": eff,
        "firing": mean_f,
        "firing_lesion": mean_l,
        "scale": sc,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--train-seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(args.seed)
    toks = torch.randint(
        0, V, (args.train_seq, args.seq), generator=torch.Generator().manual_seed(5)
    ).to(dev)

    res: dict = {"pop": args.pop, "gens": args.gens, "hand": HAND, "history": []}

    hand = evaluate(HAND, 0, dev, toks)
    res["hand_eval"] = hand
    print(
        f"  手调基因组：适应度 {hand['fitness']:.3f}  可行 {hand['viable']:.0f} "
        f"抗损伤 {hand['damage_ok']:.0f}  发放 {hand['firing']:.4f} "
        f"损伤后 {hand['firing_lesion']:.4f}",
        flush=True,
    )

    pop = [[float(x) for x in torch.rand(len(GENES), generator=gen)] for _ in range(args.pop)]
    best_overall = None
    for gi in range(args.gens):
        scored = []
        for u in pop:
            r = evaluate(decode(u), 0, dev, toks)
            scored.append((r["fitness"] if r else -1.0, u, r))
        scored.sort(key=lambda z: -z[0])
        best = scored[0]
        if best_overall is None or best[0] > best_overall[0]:
            best_overall = (best[0], best[1], best[2])
        res["history"].append(
            {"gen": gi, "best": best[0], "mean": sum(s[0] for s in scored) / len(scored)}
        )
        print(
            f"  第 {gi} 代：最优 {best[0]:.3f}  平均 {sum(s[0] for s in scored) / len(scored):.3f}",
            flush=True,
        )
        # 精英 + 锦标赛 + 变异
        elites = [list(s[1]) for s in scored[:4]]
        children = []
        while len(children) < args.pop - 4:
            picks = [
                scored[int(torch.randint(0, len(scored), (1,), generator=gen))] for _ in range(3)
            ]
            cand = max(picks, key=lambda z: z[0])
            child = [
                min(max(cand[1][i] + float(torch.randn(1, generator=gen)) * 0.10, 0.0), 1.0)
                for i in range(len(GENES))
            ]
            children.append(child)
        pop = elites + children

    res["best"] = {
        "fitness": best_overall[0],
        "unit": best_overall[1],
        "genes": decode(best_overall[1]),
        "eval": best_overall[2],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    g0 = hand is not None and math.isfinite(hand["fitness"])
    print(f"  G-0 手调基因组适应度有限：{hand['fitness']:.3f} => {'过' if g0 else '不过 => 作废'}")
    if not g0:
        return 0
    hist = res["history"]
    g1 = hist[-1]["best"] > hist[0]["best"] + 1e-6
    print(
        f"  G-1 进化有效：第 0 代 {hist[0]['best']:.3f} → 第 {len(hist) - 1} 代 {hist[-1]['best']:.3f} => {'过' if g1 else '不过'}"
    )
    bd = best_overall[2]["damage_ok"] if best_overall[2] else 0.0
    g2 = bd > hand["damage_ok"]
    print(
        f"  G-2 打败手调（抗损伤）：进化 {bd:.0f} vs 手调 {hand['damage_ok']:.0f} => {'过' if g2 else '不过'}"
    )
    print(f"  最优基因组：{ {k: round(v, 5) for k, v in decode(best_overall[1]).items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

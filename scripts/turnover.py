"""受调控的细胞更替：活动依赖的死亡 + 群体活动不足时的诞生。

预注册：`research/sel/prereg-turnover-v1.md`

```
死亡：f_i 的块均活跃 < θ_dead        ⇒ 永久静默
诞生：群体活动 < f_target 持续 K 块   ⇒ 复活一个死单元（入边重随机、增益 ×g_new）
```

生物学对应：活动依赖的存活（营养因子）+ 新生神经元的关键期超兴奋。
"""

from __future__ import annotations

# ruff: noqa: N806, N803
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.bio_all import make_task
from scripts.instrument import FIRING_WINDOW, scale_grid

OUT = _pathlib.Path("sweep_results/turnover")
L, R, V, DEG = 32, 8, 64, 8
N = L * L


def build_net(seed: int, dev: str):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.arange(N)
    x = (idx % L).float().to(dev)
    y = (idx // L).float().to(dev)
    ax = torch.arange(-R, R + 1).float()
    ox, oy = torch.meshgrid(ax, ax, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev), oy.reshape(-1).to(dev)
    C = ox.numel()
    cx = (x.unsqueeze(1) + ox.unsqueeze(0)) % L
    cy = (y.unsqueeze(1) + oy.unsqueeze(0)) % L
    d = torch.sqrt((ox.unsqueeze(0) ** 2 + oy.unsqueeze(0) ** 2).expand(N, C))
    w = torch.exp(-d / 4.0)
    w[:, (ox == 0) & (oy == 0)] = 0.0
    score = w * (1.0 + 0.5 * torch.rand(N, C, generator=gen).to(dev))
    idx_top = score.topk(DEG, dim=1).indices
    lin = (cy * L + cx).long()
    pre = lin.gather(1, idx_top)
    wrec = torch.rand(N, DEG, generator=gen).to(dev)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    w_in = (torch.randn(V, N, generator=gen) / N**0.5).to(dev)
    return w_in, pre, wrec


@torch.no_grad()
def run_block(w_in, pre, wrec, toks, dev, alive, gain, blocks=4, T=48):
    """跑 `blocks` 个块，返回 `(每神经元块均活跃, 总体发放)`。"""
    b, Tt = toks.shape
    v = torch.zeros((b, N), device=dev)
    z = torch.zeros((b, N), device=dev)
    ref = torch.zeros((b, N), device=dev)
    per_neuron = torch.zeros(N, device=dev)
    total = 0.0
    steps = 0
    for t in range(min(Tt, blocks * T)):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        cur = w_in[toks[:, t]] * gain.unsqueeze(0)
        nv = 0.95 * v + cur + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        per_neuron += z.mean(0)
        total += float(z.sum() / (alive.sum() * b))
        steps += 1
    return per_neuron / max(steps, 1), total / max(steps, 1)


@torch.no_grad()
def develop(
    w_in, pre, wrec, toks, dev, alive, gain, q_dead, rho_target, g_new, rounds=6, seed=12345
):
    """跑若干轮「跑块 → 判死亡 → 判诞生」。

    **死亡用相对阈值**（竞争有限营养因子：活跃最低的 `q_dead` 比例凋亡），
    **群体规模维持在 `rho_target`**（目标支配量）。
    绝对阈值会让整个群体灭绝 —— 第一次跑就栽在这（存活比例掉到 1%）。
    """
    gen = torch.Generator().manual_seed(seed)
    for _ in range(rounds):
        act, _ = run_block(w_in, pre, wrec, toks, dev, alive, gain)
        live = alive > 0.5
        if int(live.sum()) > 2:
            thr = torch.quantile(act[live], q_dead)
            die = (act <= thr) & live
            alive = torch.where(die, torch.zeros_like(alive), alive)
        n_need = round(rho_target * N) - int((alive > 0.5).sum())
        dead_idx = (alive < 0.5).nonzero().squeeze(1)
        if n_need > 0 and dead_idx.numel() > 0:
            perm = torch.randperm(dead_idx.numel(), generator=gen)
            pick = dead_idx[perm[: min(n_need, dead_idx.numel())]]
            alive[pick] = 1.0
            wrec[pick] = torch.rand(pick.numel(), DEG, generator=gen).to(dev)
            wrec[pick] = wrec[pick] * (1.2 / wrec[pick].sum(1, keepdim=True).clamp(min=1e-9))
            gain[pick] = g_new
        gain = 1.0 + (gain - 1.0) * 0.5  # 关键期衰减
    return alive, gain


def n_viable(w_in, pre, wrec, toks, dev, alive, gain, scales):
    lo, hi = FIRING_WINDOW
    cnt = 0
    for s in scales:
        _, fr = run_block(w_in * s, pre, wrec, toks, dev, alive, gain, blocks=1)
        if lo <= fr <= hi:
            cnt += 1
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--rho", type=float, default=0.80)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(x) for x in args.seeds.split(",")]
    scales = scale_grid(0.4, 12.0, 1.15)
    res: dict = {"cells": {}}

    for arm in ("static", "turnover"):
        vi_after, alive_frac = [], []
        for sd in seeds:
            toks, _ = make_task(args.n_seq, args.seq, 8, 10 + sd, dev)
            w_in, pre, wrec = build_net(sd, dev)
            gen = torch.Generator().manual_seed(7000 + sd)
            alive = (torch.rand(N, generator=gen) >= 0.5).float().to(dev)  # 先切一半
            gain = torch.ones(N, device=dev)
            if arm == "turnover":
                alive, gain = develop(
                    w_in,
                    pre,
                    wrec,
                    toks,
                    dev,
                    alive,
                    gain,
                    q_dead=0.10,
                    rho_target=args.rho,
                    g_new=3.0,
                    seed=7000 + sd,
                )
            vi_after.append(n_viable(w_in, pre, wrec, toks, dev, alive, gain, scales))
            alive_frac.append(float(alive.mean()))
            del w_in, pre, wrec, toks
            torch.cuda.empty_cache()
        res["cells"][arm] = {
            "viable": vi_after,
            "mean": sum(vi_after) / len(vi_after),
            "alive_frac": alive_frac,
        }
        print(
            f"  {arm:9s} 切半后 n_viable {vi_after}  存活比例 {[round(x, 2) for x in alive_frac]}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = res["cells"]
    t0 = cc["static"]["mean"] > 0
    print(f"  T-0 static 基线 {cc['static']['mean']:.1f} => {'过' if t0 else '不过 => 作废'}")
    if not t0:
        return 0
    t1 = cc["turnover"]["mean"] > cc["static"]["mean"]
    print(
        f"  T-1 更替能恢复群体：{cc['turnover']['mean']:.1f} vs static {cc['static']['mean']:.1f} "
        f"=> {'过' if t1 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

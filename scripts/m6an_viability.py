"""M6an：**自维持边界** —— 给系统一个它必须自己维持的存续量。

## 用户 2026-11-17 的更正

> 我认为它比正常生命脆的多，因为它不仅完全依赖电力，其载体电脑相比于生物也很脆弱。

对。我量的"脆性"只是**组织层**（切除后能力掉不掉）；真实生物在组织层也很韧。
**关键差别不在"有多脆"，而在"谁来维持"**：
生物的组织再韧，载体也是它自己维持的；我们的载体由宿主维持，模型完全不管。

⇒ **真正缺的是 M1（自维持边界），不是脆性。**
今晚两条看似矛盾的观察（组织层不脆 + 载体层极脆）由此统一：
**这个模型没有边界可维持。**

## 机制

一个存续量 `V ∈ [0,1]`：

```
V ← V + r·(1 − V) − c·|φ − φ*|        # 发放偏离目标就衰减，接近就恢复
驱动 ← 驱动 · clip(V, 0, 1)             # **V 直接乘在起搏与输入上**
V < V_dead ⇒ 该个体死亡（活动归零）
```

**死亡螺旋**是这个机制的定义性特征：偏离 → V 掉 → 驱动掉 → 偏得更多。
完好的网络维持在目标附近；**一旦受损到维持不住，它自己死。**

## 判据

- **AN-0 完好网络能维持** ★：`V` 在 T=2048 步后仍 > 0.5（否则机制不工作，作废）；
- **AN-1 受损会死** ★★：切 50% 后 `V` 终值 < 0.5（**这就是"脆"**）；
- **AN-2 死亡螺旋** ：切后死亡个体比例 > 完好时的比例。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools, run_spont

OUT = _pathlib.Path("sweep_results/m6an_viability")
K = 4
T_HIST, T_SPONT = 1024, 2048
R_REC = 0.10
C_COST = 3.0
V_DEAD = 0.15


@torch.no_grad()
def run_alive(
    net,
    seed: int,
    hist,
    dev: str,
    t_spont: int,
    alive=None,
    use_v: bool = True,
    phi_star: float = 0.08,
):
    """带存续量的自发期。`use_v=False` 时退化为普通 run_spont 的动力学。"""
    w_in, pre, wrec, _pid = net
    w_in = w_in * 0.7
    B = hist.shape[0]
    alive = torch.ones(N, device=dev) if alive is None else alive
    g = torch.Generator().manual_seed(seed + 31337)
    pace = ((torch.rand(N, generator=g) < 0.02).float() * 0.5).to(dev)
    v = torch.zeros((B, N), device=dev)
    z = torch.zeros((B, N), device=dev)
    ref = torch.zeros((B, N), device=dev)
    f_run = 0.08
    # **run_spont 在历史期会更新 gain（eta_hist=2.0）**；漏了它历史轨迹就不同
    gain = torch.ones(N, device=dev)
    f_i = torch.full((N,), 0.08, device=dev)
    V = torch.ones(B, device=dev)
    phi_slow = (
        phi_star.clone().to(dev)
        if isinstance(phi_star, torch.Tensor)
        else torch.full((B,), phi_star, device=dev)
    )

    for t in range(hist.shape[1]):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi = float(z.mean())
        cur = w_in[hist[:, t]] * gain.unsqueeze(0) - 1.0 * (f_run / 0.08) * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        nz = torch.where(ref > 0, torch.zeros_like(nv), ((nv - 0.5) / 0.5 > 0).float()) * alive
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        f_i = 0.9 * f_i + 0.1 * z.mean(0)
        if t % 4 == 3:
            gain = torch.clamp(gain * torch.exp(2.0 * (0.08 - f_i)), 0.2, 5.0)
        f_run = 0.9 * f_run + 0.1 * phi

    rates = torch.zeros((B, N), device=dev)
    for t in range(t_spont):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi_t = z.mean(1) if use_v else None
        # **环路增益必须 < 1**：V 若乘在全部驱动上，任何偏离都被放大成死亡螺旋，
        # 完好网络也活不下来（第一版三条种子全死）。只让它影响 80%。
        vf = (0.2 + 0.8 * V.clamp(0.0, 1.0)) if use_v else torch.ones(B, device=dev)
        drive = pace.unsqueeze(0) * vf.unsqueeze(1)
        phi = float(z.mean())
        cur = drive - 1.0 * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        nz = torch.where(ref > 0, torch.zeros_like(nv), ((nv - 0.5) / 0.5 > 0).float()) * alive
        if use_v:
            # V 掉到阈值以下 ⇒ **真正判死**（活动归零，不可恢复）
            nz = nz * (V >= V_DEAD).float().unsqueeze(1)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        if use_v:
            # **存续量积分的是慢变量**：逐时刻发放波动极大，直接用会把成本项变成噪声
            # （第一版就是这么死的：完好网络 V=0.000）。代谢量在生物上也是积分量。
            phi_slow = 0.995 * phi_slow + 0.005 * phi_t
            V = (V + R_REC * (1.0 - V) - C_COST * (phi_slow - phi_star).abs()).clamp(0.0, 1.0)
        if t >= t_spont - 64:
            rates += z
    return rates / 64, V


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--per", type=int, default=32)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": []}
    for sd in range(args.seeds):
        net = build_pools(sd, dev, K, 0.0)
        toks = make_family_hist(0, sd * 71 + 3, T_HIST, args.per, dev)
        # **目标值必须标定到网络自己的工作点**，不能拍。第一版写死 0.08，
        # 而网络工作在 0.15 ⇒ 偏离 0.07×成本 6.0 每步衰减 0.42 > 恢复 0.05 ⇒ 完好网络也死。
        r0 = run_spont(
            sd, 0.5, 0, 0.7, 0.02, dev, K, 0.0, 0.0, -1.0, hist_in=toks, t_spont=T_SPONT, net=net
        )[0]
        # **逐个体目标**：批次均值会让个体间的正常差异被当成"偏离"
        phi_star = r0.mean(1)
        _, v_intact = run_alive(net, sd, toks, dev, T_SPONT, phi_star=phi_star)
        g = torch.Generator().manual_seed(sd + 311)
        mask = (torch.randperm(N, generator=g).to(dev) >= int(args.frac * N)).float()
        _, v_les = run_alive(net, sd, toks, dev, T_SPONT, alive=mask, phi_star=phi_star)
        row = {
            "seed": sd,
            "phi_star": float(phi_star.mean()) if isinstance(phi_star, torch.Tensor) else phi_star,
            "v_intact": float(v_intact.mean()),
            "v_lesion": float(v_les.mean()),
            "dead_intact": float((v_intact < V_DEAD).float().mean()),
            "dead_lesion": float((v_les < V_DEAD).float().mean()),
        }
        res["cells"].append(row)
        print(
            f"  seed {sd}: 完好 V={row['v_intact']:.3f}（死亡 {row['dead_intact']:.2f}）  "
            f"切 50% V={row['v_lesion']:.3f}（死亡 {row['dead_lesion']:.2f}）",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )

    print("")
    print("判：")
    rows = res["cells"]
    m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
    an0 = m("v_intact") > 0.5
    print(
        f"  AN-0 完好网络能维持（V>0.5）：{m('v_intact'):.3f} => {'过' if an0 else '不过 => 作废'}"
    )
    if not an0:
        return 0
    an1 = m("v_lesion") < 0.5
    print(f"  AN-1 受损会死（V<0.5）：{m('v_lesion'):.3f} => {'过（有脆性）' if an1 else '不过'}")
    an2 = m("dead_lesion") > m("dead_intact")
    print(
        f"  AN-2 死亡螺旋：切后死亡比例 {m('dead_lesion'):.2f} vs 完好 {m('dead_intact'):.2f} "
        f"=> {'过' if an2 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

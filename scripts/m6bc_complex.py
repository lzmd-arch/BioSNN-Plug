"""M6bc：**工作点在复平面上吗**（预注册 `research/sel/prereg-complex-working-point-v1.md`）

## 一句话

实值网络的共同模态与个体差异**争同一个自由度**（振幅）；复值网络多一个**相位**，
于是「整合住相位锁定、分化住幅度与相位散布」成为可能——这就是**频率复用**。

## 引擎

```
z_n ← α_n·z_n + scale·Σ_m W_nm·φ(z_m) + I_n
α_n = leak·exp(i·ω_n)        ω_n ~ N(0, σ_ω)   ← 异质固有频率
φ(z) = z·1[|z|>θ]            发放 = |z| 越过 θ
复位：z ← z·(1 − θ/|z|)      **保相位**减去阈值（复数版 v−θ）
```

`σ_ω = 0` 时 α 退化为实数，应复现实值引擎（C-0 仪器自检）。

## 决定性控制（C-2）

**相位打乱**：对每个（样本，神经元）独立随机旋转 `z ← z·e^{iψ}`，**保留 |z|**。
- 掉幅大 ⇒ 信息在相位里（相空间效应）；
- 掉幅为零 ⇒ 复值只是「双倍实数维度」（维度效应）。

## 剂量

`σ_ω ∈ {0, 0.02, 0.1, 0.3}`。**种子 ≥ 6。**
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist, spatial_bins
from scripts.instrument import majority_baseline, mlp_probe, probe
from scripts.m6_pools import N, build_pools
from scripts.m6v_dimension import pr_of

OUT = _pathlib.Path("sweep_results/m6bc_complex")
K, SIDE = 4, 32
T_HIST, T_SPONT = 1024, 2048
FAMS = list(range(8))
P_LONG = 0.5
LEAK, THETA, SCALE, G_FIELD = 0.95, 0.5, 0.7, 1.0
T_LATE = 512


@torch.no_grad()
def run_complex(
    seed: int,
    dev: str,
    hist_in: torch.Tensor,
    *,
    sigma_omega: float = 0.0,
    hetero: float = 0.0,
    t_spont: int = T_SPONT,
    net=None,
    leak: float = LEAK,
    theta: float = THETA,
    scale: float = SCALE,
    g_field: float = G_FIELD,
    record_len: int = T_LATE,
):
    """复值 LIF。返回 `(发放率 (B,N), 复状态均值 (B,N) complex)`。"""
    w_in, pre, wrec, _pid = (
        net if net is not None else build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    )
    w_in = w_in * scale
    g = torch.Generator().manual_seed(seed + 60601)
    omega = torch.randn(N, generator=g).to(dev) * sigma_omega
    rg = torch.Generator().manual_seed(seed + 60602)
    amp_het = (1.0 + hetero * torch.randn(N, generator=rg).to(dev)).clamp(0.5, 1.5)
    cos_w, sin_w = torch.cos(omega) * amp_het, torch.sin(omega) * amp_het

    B, T = hist_in.shape
    x = torch.zeros((B, N), device=dev)
    y = torch.zeros((B, N), device=dev)
    ref = torch.zeros((B, N), device=dev)
    stats = []

    for t in range(T + t_spont):
        mag = torch.sqrt(x * x + y * y)
        fired = (mag > theta) & (ref <= 0)
        # **单位复脉冲**：幅度 1、相位 = 膜电位相位。实值版的 0/1 脉冲是它的 ω=0 特例。
        inv = 1.0 / mag.clamp(min=1e-9)
        zx = x * inv * fired
        zy = y * inv * fired
        rec_x = torch.zeros_like(x)
        rec_y = torch.zeros_like(y)
        for j in range(pre.shape[1]):
            rec_x = rec_x + zx[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
            rec_y = rec_y + zy[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi_g = float(fired.float().mean())
        # 复数泄漏 = 旋转 + 衰减；输入是实的（电流加在实部）
        nx = leak * (cos_w * x - sin_w * y) + rec_x - g_field * phi_g
        ny = leak * (sin_w * x + cos_w * y) + rec_y
        if t < T:
            cur = w_in[hist_in[:, t]]
            nx = nx + cur
        # **保相位复位**：沿径向减去阈值
        mag2 = torch.sqrt(nx * nx + ny * ny).clamp(min=1e-9)
        scale_r = (1.0 - theta / mag2).clamp(min=0.0)
        nf = (mag2 > theta) & (ref <= 0)
        nx = torch.where(nf, nx * scale_r, nx)
        ny = torch.where(nf, ny * scale_r, ny)
        ref = torch.clamp(ref + 2.0 * nf.float() - 1.0, 0.0, 2.0)
        x, y = nx, ny
        if t >= T + t_spont - record_len:
            stats.append(torch.complex(x, y))

    z = torch.stack(stats, 2)  # (B, N, record_len)
    rate = (z.abs() > theta * 0.5).float().mean(2)
    return rate, z.mean(2)


def split_features(zmean: torch.Tensor, binned: bool = False) -> torch.Tensor:
    """复状态 → 实特征 `[Re, Im]`。`binned` 时两路各按空间块池化后再拼。"""
    if not binned:
        return torch.cat([zmean.real, zmean.imag], 1)
    return torch.cat(
        [spatial_bins(zmean.real.contiguous()), spatial_bins(zmean.imag.contiguous())], 1
    )


def scramble_phase(zmean: torch.Tensor, seed: int) -> torch.Tensor:
    """**C-2 控制**：每个（样本，神经元）独立随机旋转，**保留 |z|**。"""
    g = torch.Generator().manual_seed(seed + 313)
    ang = torch.rand(zmean.shape, generator=g) * (2.0 * torch.pi)
    rot = torch.polar(torch.ones_like(zmean.real, dtype=torch.float32), ang.to(zmean.device))
    return zmean * rot


def shape_stats(zmean: torch.Tensor) -> dict:
    S = split_features(zmean)
    h = (torch.arange(N) % SIDE >= 16).long()
    h2 = torch.cat([h, h]).bool()  # 实部/虚部各有一份同样的半球划分
    A, B = S[:, ~h2], S[:, h2]
    a_, b_ = A.mean(1) - A.mean(), B.mean(1) - B.mean()
    integ = abs(float((a_ * b_).sum() / (a_.norm() * b_.norm() + 1e-12)))
    segs = [float((H.std(1) / (H.mean(1) + 1e-9)).mean()) for H in (A, B)]
    return {"pr": pr_of(S), "integ": integ, "seg": sum(segs) / 2}


@torch.no_grad()
def cell(seed: int, sigma_omega: float, hetero: float, per_fam: int, dev: str) -> dict:
    net = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    Z, FR, Y, BIN = [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 809 + fam * 43 + 21, T_HIST, per_fam, dev)
        rate, zm = run_complex(seed, dev, toks, sigma_omega=sigma_omega, hetero=hetero, net=net)
        Z.append(zm)
        FR.append(float(rate.mean()))
        BIN.append(split_features(zm, binned=True))
        Y.append(torch.full((zm.shape[0],), fam, device=zm.device))
    Z = torch.cat(Z)
    Yc = torch.cat(Y)
    out = {
        "firing": sum(FR) / len(FR),
        "BIN": torch.cat(BIN),
        "Y": Yc,
        **shape_stats(Z),
    }
    out["FULL"] = split_features(Z)
    out["FULLscr"] = split_features(scramble_phase(Z, seed))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--sigmas", default="0,0.02,0.1,0.3")
    ap.add_argument("--hetero", type=float, default=0.0)
    ap.add_argument("--per-fam", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for sg in [float(s) for s in args.sigmas.split(",")]:
        rows = []
        for sd in range(args.seeds):
            r = cell(sd, sg, args.hetero, args.per_fam, dev)
            perm = torch.randperm(r["Y"].shape[0], generator=torch.Generator().manual_seed(51 + sd))
            cut = int(r["Y"].shape[0] * 0.75)
            Yp = r["Y"][perm]
            r["linear"] = probe(r["BIN"][perm], Yp, cut, len(FAMS))[0]
            r["mlp"] = mlp_probe(r["FULL"][perm], Yp, cut, len(FAMS))[0]
            r["mlp_scr"] = mlp_probe(r["FULLscr"][perm], Yp, cut, len(FAMS))[0]
            r["floor"] = majority_baseline(Yp[:cut], Yp[cut:], len(FAMS))
            for k in ("Z", "FULL", "BIN", "FULLscr", "Y"):
                r.pop(k, None)
            rows.append(r)
            print(
                f"  σ_ω={sg:g} seed {sd}: PR {r['pr']:6.2f} 整合 {r['integ']:.3f} "
                f"分离 {r['seg']:.2f} 线性 {r['linear']:.3f} MLP {r['mlp']:.3f} "
                f"打乱 {r['mlp_scr']:.3f} 发放 {r['firing']:.4f}",
                flush=True,
            )
            if args.smoke:
                break
        if not rows:
            continue
        v = {k: sum(x[k] for x in rows) / len(rows) for k in rows[0]}
        v["rows"] = rows
        v["sigma_omega"] = sg
        v["phase_gain"] = v["mlp"] - v["mlp_scr"]
        res["cells"][f"sigma={sg:g}"] = v
        print(
            f"  => σ_ω={sg:g}: PR {v['pr']:.2f} 整合 {v['integ']:.3f} 分离 {v['seg']:.2f} "
            f"可读 {max(v['linear'], v['mlp']):.3f}（底板 {v['floor']:.3f}）"
            f" 相位增益 {v['phase_gain']:+.3f} 发放 {v['firing']:.4f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    c = res["cells"]
    c0 = c.get("sigma=0")
    if c0 is not None:
        ok0 = 0.01 <= c0["firing"] <= 0.20 and max(c0["linear"], c0["mlp"]) > c0["floor"] + 0.15
        print(f"  C-0 仪器自检（σ_ω=0 复现工作点）：{'过' if ok0 else '**不过 ⇒ 作废**'}")
        if not ok0:
            return 0
    hit = [
        k
        for k, v in c.items()
        if v["sigma_omega"] > 0
        and 0.01 <= v["firing"] <= 0.20
        and v["pr"] > 10
        and v["integ"] > 0.3
        and v["seg"] > 0.5
        and max(v["linear"], v["mlp"]) > v["floor"] + 0.15
    ]
    print(f"  C-1 存在复值工作点：{hit or '无'} => {'过' if hit else '不过'}")
    if hit:
        g = max(c[k]["phase_gain"] for k in hit)
        print(
            f"  C-2 相位承载信息（打乱后掉幅 >0.15）：最大相位增益 {g:+.3f} => "
            f"{'过：相空间效应' if g > 0.15 else ('方向相反：只有维度效应' if g < 0.05 else '中间，不判方向')}"
        )
    if not hit:
        print("  C-4 全负 ⇒ **工作点不在复平面**，照登负结果，不调剂量重跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from scripts.instrument import FIRING_WINDOW, majority_baseline, mlp_probe, probe
from scripts.m6_pools import N, build_pools, run_spont
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
    """复值 LIF。返回 `(发放率 (B,N), 复状态均值 (B,N) complex)`。

    **与 `run_spont` 同构**：脉冲是上一步的量、复位是"减去上一步脉冲的 θ 倍"、
    阈值判在新膜电位上、稳态增益与起搏电流照搬。ω=0 时应当逐项对上实值引擎。
    """
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
    px = torch.zeros((B, N), device=dev)  # 复脉冲（上一步）：幅度 0/1、相位 = 膜电位相位
    py = torch.zeros((B, N), device=dev)
    ref = torch.zeros((B, N), device=dev)
    gain = torch.ones(N, device=dev)
    f_i = torch.full((N,), 0.08, device=dev)
    f_run = 0.08
    pg = torch.Generator().manual_seed(seed + 31337)
    pace = ((torch.rand(N, generator=pg) < 0.02).float() * 0.5).to(dev)
    stats = []
    rate_acc = torch.zeros((B, N), device=dev)

    for t in range(T + t_spont):
        rec_x = torch.zeros_like(x)
        rec_y = torch.zeros_like(y)
        for j in range(pre.shape[1]):
            rec_x = rec_x + px[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
            rec_y = rec_y + py[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi_g = float(px.abs().mean())
        nx = leak * (cos_w * x - sin_w * y) + rec_x - theta * px - g_field * (f_run / 0.08) * phi_g
        ny = leak * (sin_w * x + cos_w * y) + rec_y - theta * py
        # **起搏只在自发期**：历史期由输入驱动，自发期靠 2% 起搏神经元维持
        drive = w_in[hist_in[:, t]] * gain.unsqueeze(0) if t < T else pace.unsqueeze(0)
        nx = nx + drive
        # **发放判在实部**（实部 = 电压，虚部 = 正交分量）。
        # 用 |z|>θ 会让负膜电位也触发，而复位沿相位方向 ⇒ 等于把电流加回去。
        mag = torch.sqrt(nx * nx + ny * ny)
        nf = (nx > theta) & (ref <= 0)
        inv = 1.0 / mag.clamp(min=1e-9)
        px, py = nx * inv * nf, ny * inv * nf
        ref = torch.clamp(ref + 2.0 * nf.float() - 1.0, 0.0, 2.0)
        x, y = nx, ny
        f_run = 0.9 * f_run + 0.1 * float(nf.float().mean())
        if t < T:
            f_i = 0.9 * f_i + 0.1 * nf.float().mean(0)
            if t % 4 == 3:
                gain = torch.clamp(gain * torch.exp(2.0 * (0.08 - f_i)), 0.2, 5.0)
        if t >= T + t_spont - record_len:
            # **记录脉冲相量**（幅度 0/1、相位 = 发放时的膜电位相位）。
            # 记膜电位均值是错的：ω=0 时它是一堆常数，解码会掉到地板。
            stats.append(torch.complex(px, py))
            rate_acc += nf.float()

    z = torch.stack(stats, 2)  # (B, N, record_len)
    return rate_acc / max(record_len, 1), z.mean(2)


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


def shape_stats(rate: torch.Tensor) -> dict:
    """三个形状量**都算在脉冲率上** —— 与 `m6az` 的实值工作点可以直接比。"""
    h = (torch.arange(N) % SIDE >= 16).long()
    A, B = rate[:, h == 0], rate[:, h == 1]
    a_, b_ = A.mean(1) - A.mean(), B.mean(1) - B.mean()
    integ = abs(float((a_ * b_).sum() / (a_.norm() * b_.norm() + 1e-12)))
    segs = [float((H.std(1) / (H.mean(1) + 1e-9)).mean()) for H in (A, B)]
    return {"pr": pr_of(rate), "integ": integ, "seg": sum(segs) / 2}


def phase_locking(phasor: torch.Tensor) -> float:
    """**相位锁定值** R = |平均单位相量|。1 = 全群同相，0 = 相位均匀散布。"""
    ang = torch.angle(phasor)
    mag = phasor.abs()
    w = mag / mag.sum(1, keepdim=True).clamp(min=1e-9)
    return float(
        (w * torch.cos(ang)).sum(1).pow(2).add((w * torch.sin(ang)).sum(1).pow(2)).sqrt().mean()
    )


@torch.no_grad()
def real_cell(seed: int, per_fam: int, dev: str) -> dict:
    """**实值基线**（同一批族、同参数）。C-1 的对照就是它，不是"另一档旋钮"。"""
    net = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    R, FR, Y, BIN = [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 809 + fam * 43 + 21, T_HIST, per_fam, dev)
        r = run_spont(
            seed,
            0.5,
            0,
            0.7,
            0.02,
            dev,
            K,
            0.0,
            0.0,
            -1.0,
            hist_in=toks,
            t_spont=T_SPONT,
            net=net,
        )[0]
        R.append(r)
        FR.append(float(r.mean()))
        BIN.append(spatial_bins(r.contiguous()))
        Y.append(torch.full((r.shape[0],), fam, device=r.device))
    S = torch.cat(R)
    Yc = torch.cat(Y)
    perm = torch.randperm(Yc.shape[0], generator=torch.Generator().manual_seed(51 + seed))
    cut = int(Yc.shape[0] * 0.75)
    return {
        "firing": sum(FR) / len(FR),
        "linear": probe(torch.cat(BIN)[perm], Yc[perm], cut, len(FAMS))[0],
        "mlp": mlp_probe(S[perm], Yc[perm], cut, len(FAMS))[0],
        "floor": majority_baseline(Yc[perm][:cut], Yc[perm][cut:], len(FAMS)),
        **shape_stats(S),
    }


@torch.no_grad()
def cell(seed: int, sigma_omega: float, hetero: float, per_fam: int, dev: str) -> dict:
    net = build_pools(seed, dev, K, 0.0, p_long=P_LONG)
    Z, R, FR, Y, BIN = [], [], [], [], []
    for fam in FAMS:
        toks = make_family_hist(fam, seed * 809 + fam * 43 + 21, T_HIST, per_fam, dev)
        rate, zm = run_complex(seed, dev, toks, sigma_omega=sigma_omega, hetero=hetero, net=net)
        Z.append(zm)
        R.append(rate)
        FR.append(float(rate.mean()))
        BIN.append(split_features(zm, binned=True))
        Y.append(torch.full((zm.shape[0],), fam, device=zm.device))
    Z = torch.cat(Z)
    Yc = torch.cat(Y)
    out = {
        "firing": sum(FR) / len(FR),
        "BIN": torch.cat(BIN),
        "Y": Yc,
        "lock": phase_locking(Z),
        **shape_stats(torch.cat(R)),
    }
    out["FULL"] = split_features(Z)
    out["FULLscr"] = split_features(scramble_phase(Z, seed))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--sigmas", default="0,0.02,0.1,0.3")
    ap.add_argument("--hetero", type=float, default=0.0)
    ap.add_argument("--per-fam", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    n_seeds = 1 if args.smoke else args.seeds

    print("实值基线（A/B 的对照）：", flush=True)
    real = [real_cell(sd, args.per_fam, dev) for sd in range(n_seeds)]
    for sd, r in enumerate(real):
        print(
            f"  seed {sd}: 发放 {r['firing']:.4f} 线性 {r['linear']:.3f} "
            f"MLP {r['mlp']:.3f} PR {r['pr']:6.2f} 底板 {r['floor']:.3f}",
            flush=True,
        )
    mean = lambda rows, k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
    base_read = (
        mean(real, "linear") if mean(real, "linear") > mean(real, "mlp") else mean(real, "mlp")
    )
    base_f, base_pr = mean(real, "firing"), mean(real, "pr")
    print(f"  => 基线 可读 {base_read:.3f} 发放 {base_f:.4f} PR {base_pr:.2f}", flush=True)

    res: dict = {"real": real, "cells": {}}
    for sg in [float(x) for x in args.sigmas.split(",")]:
        rows = []
        for sd in range(n_seeds):
            r = cell(sd, sg, args.hetero, args.per_fam, dev)
            perm = torch.randperm(r["Y"].shape[0], generator=torch.Generator().manual_seed(51 + sd))
            cut = int(r["Y"].shape[0] * 0.75)
            Yp = r["Y"][perm]
            r["linear"] = probe(r["BIN"][perm], Yp, cut, len(FAMS))[0]
            r["mlp"] = mlp_probe(r["FULL"][perm], Yp, cut, len(FAMS))[0]
            r["mlp_scr"] = mlp_probe(r["FULLscr"][perm], Yp, cut, len(FAMS))[0]
            r["floor"] = majority_baseline(Yp[:cut], Yp[cut:], len(FAMS))
            r["read"] = max(r["linear"], r["mlp"])
            r["phase_gain"] = r["mlp"] - r["mlp_scr"]
            for k in ("Z", "FULL", "BIN", "FULLscr", "Y"):
                r.pop(k, None)
            rows.append(r)
            print(
                f"  σ_ω={sg:g} seed {sd}: PR {r['pr']:6.2f} 整合 {r['integ']:.3f} "
                f"分离 {r['seg']:.2f} 线性 {r['linear']:.3f} MLP {r['mlp']:.3f} "
                f"打乱 {r['mlp_scr']:.3f} 锁相 {r['lock']:.3f} 发放 {r['firing']:.4f}",
                flush=True,
            )
        v = {k: mean(rows, k) for k in rows[0]}
        v["rows"] = rows
        v["sigma_omega"] = sg
        res["cells"][f"sigma={sg:g}"] = v
        print(
            f"  => σ_ω={sg:g}: 可读 {v['read']:.3f}（基线 {base_read:.3f}，"
            f"差 {v['read'] - base_read:+.3f}）相位增益 {v['phase_gain']:+.3f} "
            f"PR {v['pr']:.2f}（基线 {base_pr:.2f}）发放 {v['firing']:.4f}",
            flush=True,
        )
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判（照 amend1 的相对 A/B）：")
    c = res["cells"]
    c0 = c.get("sigma=0")
    fw = FIRING_WINDOW
    if c0 is not None:
        a = abs(c0["firing"] - base_f) <= 0.05
        b = c0["read"] > c0["floor"] + 0.05
        print(
            f"  C-0 仪器自检：|发放差| {abs(c0['firing'] - base_f):.4f} ≤0.05 {a}；"
            f"可读 {c0['read']:.3f} > 底板+0.05 {c0['floor'] + 0.05:.3f} {b} => "
            f"{'过' if a and b else '**不过 ⇒ 作废**'}"
        )
        if not (a and b):
            return 0
    hit = [
        k
        for k, v in c.items()
        if v["sigma_omega"] > 0 and fw[0] <= v["firing"] <= fw[1] and v["read"] - base_read >= 0.15
    ]
    print(f"  C-1 相位带来可读性（≥ 实值基线 +0.15）：{hit or '无'} => {'过' if hit else '不过'}")
    if hit:
        g = max(c[k]["phase_gain"] for k in hit)
        print(
            f"  C-2 信息在相位里（可读 − 相位打乱 ≥0.15）：最大 {g:+.3f} => "
            f"{'过' if g >= 0.15 else ('方向相反：只是双倍维度' if g < 0.05 else '中间，不判方向')}"
        )
        ok3 = all(c[k]["pr"] >= 0.8 * base_pr for k in hit)
        print(
            f"  C-3 不许用塌缩换解码（PR ≥ 0.8×基线 {0.8 * base_pr:.2f}）："
            f"{[(k, round(c[k]['pr'], 2)) for k in hit]} => {'过' if ok3 else '**不过**'}"
        )
    else:
        print("  C-4 全负 ⇒ **相位这一个自由度买不到实值读出看不见的可读性**，照登。")
    print("")
    print("  限定语：本轮默认工作点是 PR≈1 的秩 1 共同模态 ⇒ 只回答「相位能不能承载信息」，")
    print("  **不回答「三轴能不能兼得」**（那要在 amp_gate 复核值引擎里问）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

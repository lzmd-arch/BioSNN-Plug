"""复数哈密顿系统的相变是否**结构性**的。

预注册：`research/sel/prereg-quantum-phase-transition-v1.md`

## 模型（**经典地积分量子方程**，不是量子计算）

```
i dψ/dt = H ψ + g·|ψ|²ψ − i γ ψ        H = J + Γ·I
```

`J` 按 3 维折叠薄片上的距离衰减；`Γ` 横场；`g` 非线性；`γ` 显式退相干。
观测量：`|ψ_i|²`。

**不需要相干、不需要低温。** 价值在数学结构（哈密顿 → 相变 → 临界）。

## 要回答

v8 的诊断是"临界点是我手调的"。这一轮问：**相变点是否由 `J` 决定？**
若是 ⇒ 阈值从外生变内生。
"""

from __future__ import annotations

# ruff: noqa: N802, N803, N806
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

OUT = _pathlib.Path("sweep_results/quantum_phase")
L = 32
N = L * L
R = 8


def build_J(lam: float, deg: int, A: float, seed: int, dev: str):
    """按折叠薄片上的 3 维距离建耦合矩阵 `J`（稠密，N=1024）。"""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.arange(N)
    x = (idx % L).float().to(dev)
    y = (idx // L).float().to(dev)
    z = A * torch.sin(2.0 * math.pi * x / L)
    ax = torch.arange(-R, R + 1).float()
    ox, oy = torch.meshgrid(ax, ax, indexing="ij")
    ox, oy = ox.reshape(-1).to(dev), oy.reshape(-1).to(dev)
    C = ox.numel()
    cx = (x.unsqueeze(1) + ox.unsqueeze(0)) % L
    cy = (y.unsqueeze(1) + oy.unsqueeze(0)) % L
    zc = A * torch.sin(2.0 * math.pi * cx / L)
    d2 = (ox.unsqueeze(0) ** 2 + oy.unsqueeze(0) ** 2).expand(N, C)
    d = torch.sqrt(d2 + (z.unsqueeze(1) - zc) ** 2)
    w = torch.exp(-d / lam)
    w[:, (ox == 0) & (oy == 0)] = 0.0
    score = w * (1.0 + 0.5 * torch.rand(N, C, generator=gen).to(dev))
    idx_top = score.topk(deg, dim=1).indices
    lin = (cy * L + cx).long()
    pre = lin.gather(1, idx_top)
    J = torch.zeros(N, N, device=dev)
    rows = torch.arange(N, device=dev).unsqueeze(1).expand(N, deg)
    J[rows, pre] = w.gather(1, idx_top)
    return J.to(torch.complex64)


def make_U(J, dt: float = 0.05):
    """线性部分的**酉**传播子 `exp(−i·dt·J)`，只算一次。"""
    return torch.linalg.matrix_exp(-1j * dt * J)


def evolve(U, Gamma, g, gamma, psi0, drive=None, dt=0.05, steps=120):
    """Strang 分裂：半个非线性 → 整个线性（酉）→ 半个非线性 → 衰减。

    **不能用显式欧拉**：它的稳定域不含虚轴，对哈密顿系统**无条件不稳定**
    （第一次尝试就全 NaN）。分裂步 + 酉传播子保范数，不会爆。
    """
    import cmath

    psi = psi0.clone()
    lin = U * cmath.exp(-1j * dt * Gamma)
    dec = math.exp(-dt * gamma)
    for _ in range(steps):
        psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
        psi = lin @ psi
        if drive is not None:
            psi = psi + drive
        psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
        psi = psi * dec
    return psi


def sensitivity(U, Gamma, g, gamma, psi0, x, dt=0.05, steps=120):
    """弱输入引起的响应幅度（对输入范数归一）。"""
    a = evolve(U, Gamma, g, gamma, psi0, None, dt, steps)
    b = evolve(U, Gamma, g, gamma, psi0, x, dt, steps)
    d = (b.abs() ** 2 - a.abs() ** 2).norm()
    return float(d / max(float(x.norm()), 1e-9))


def scan(U, gammas_list, g, gamma, psi0, x):
    out = []
    for Gm in gammas_list:
        out.append(sensitivity(U, Gm, g, gamma, psi0, x))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--A", type=float, default=16.0)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--gammas", default="0.02,0.05,0.1,0.2,0.4,0.8,1.6,3.2")
    ap.add_argument("--jscales", default="0.5,1.0,2.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Gs = [float(s) for s in args.gammas.split(",")]

    res: dict = {
        "N": N,
        "lam": args.lam,
        "deg": args.deg,
        "A": args.A,
        "g": args.g,
        "gamma": args.gamma,
    }

    gen = torch.Generator().manual_seed(args.seed)
    psi0 = torch.randn(N, generator=gen, dtype=torch.float32)
    psi0 = (psi0 + 0j).to(dev)
    psi0 = psi0 / psi0.norm()
    x = torch.randn(N, generator=gen).to(dev) * 1e-3 + 0j

    curves = {}
    peaks = {}
    for scale in [float(s) for s in args.jscales.split(",")]:
        J = build_J(args.lam, args.deg, args.A, args.seed, dev) * scale
        U = make_U(J)
        vals = scan(U, Gs, args.g, args.gamma, psi0, x)
        finite = all(math.isfinite(v) for v in vals)
        curves[f"{scale:.2f}"] = vals
        if finite:
            k = max(range(len(vals)), key=lambda i: vals[i])
            peaks[f"{scale:.2f}"] = {"Gamma": Gs[k], "sens": vals[k], "idx": k}
        print(
            f"  J×{scale:.2f}  敏感性 {[round(v, 4) for v in vals]}"
            f"{'' if finite else '  **含非有限值**'}",
            flush=True,
        )
        torch.cuda.empty_cache()
    res["curves"] = curves
    res["peaks"] = peaks

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    ok0 = len(peaks) == len(curves)
    print(f"  Q-0 动力学不发散（无 NaN/Inf）=> {'过' if ok0 else '不过 => 作废'}")
    if not ok0:
        return 0
    best = max(peaks.values(), key=lambda p: p["sens"])
    edge = min(curves["1.00"][0], curves["1.00"][-1])
    q1 = best["sens"] > 1.5 * max(edge, 1e-12)
    print(
        f"  Q-1 相变存在：峰值 {best['sens']:.3f} vs 边界 {edge:.3f}（要 >1.5×）=> "
        f"{'过' if q1 else '不过'}"
    )
    idxs = [peaks[k]["idx"] for k in ("0.50", "1.00", "2.00") if k in peaks]
    moved = len(idxs) == 3 and idxs[0] != idxs[1] and idxs[1] != idxs[2]
    mono = len(idxs) == 3 and (idxs[0] <= idxs[1] <= idxs[2] or idxs[0] >= idxs[1] >= idxs[2])
    print(
        f"  Q-2 相变是结构的：J×0.5/1/2 的峰值位置 {idxs} => "
        f"{'过（单调移动）' if moved and mono else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""M6ay：**把量子子系统接进闭环** —— 干涉和坍缩能不能带来经典拿不到的东西。

## 用户 2026-11-17

> 接上仿真。

## 设计

12 比特量子寄存器（4096 个复振幅，8 GB 卡上很轻松）：

```
每步：① 相位门   theta_i <- omega x (网络第 i 组的发放)     网络的过去 -> 量子的相位
      ② 混合层   H 单比特 + CNOT 阶梯（固定）              干涉 + 纠缠
      ③ 测量     按 |psi|^2 采样一个比特串（真随机源，Born 规则）
      ④ 低 3 位 -> 下一个符号 -> 喂回网络                  闭环合上
```

对照：同一闭环，但用 argmax 选符号（M6ai 的经典版，实测带历史 1.07-1.38x）。

## 判据

- **AY-0** 两版都不熄火（>=80% 步在窗内）；
- **AY-1 量子版更强地带历史** (star): `d_hist/d_null > 1.5`（经典版是 1.07-1.38）；
- **AY-2 行为非平凡**：符号序列归一化熵 > 0.5。
"""

from __future__ import annotations

# ruff: noqa: N806
import argparse
import json
import math
import os
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import torch

from scripts.hist_families import make_family_hist
from scripts.m6_pools import N, build_pools

OUT = _pathlib.Path("sweep_results/m6ay_quantum_loop")
K = 4
T_HIST, T_CLOSED = 1024, 256
N_Q = 12
GROUPS = 16  # 1024 / 16 = 64，必须整除
I_PACE = 0.5


def true_seed() -> int:
    return int.from_bytes(os.urandom(8), "little") >> 1


def hadamard_layer(psi: torch.Tensor, n: int) -> torch.Tensor:
    out = psi
    for b in range(n):
        step = 1 << b
        sh = out.reshape(-1, 2 * step)
        a, c = sh[:, :step], sh[:, step:]
        sh = torch.cat([(a + c) / math.sqrt(2), (a - c) / math.sqrt(2)], dim=1)
        out = sh.reshape(-1)
    return out


def cnot_ladder(psi: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.arange(psi.numel(), device=psi.device)
    for b in range(n - 1):
        ctrl = ((idx >> b) & 1).bool()
        tgt = 1 << (b + 1)
        src = idx ^ tgt
        new = psi.clone()
        new[ctrl] = psi[src[ctrl]]
        psi = new
    return psi


@torch.no_grad()
def run_loop(net, seed: int, hist, dev: str, t_closed: int, use_quantum: bool, omega: float):
    w_in, pre, wrec, _pid = net
    w_in = w_in * 0.7
    B = 1
    v = torch.zeros((B, N), device=dev)
    z = torch.zeros((B, N), device=dev)
    ref = torch.zeros((B, N), device=dev)
    f_run = 0.08
    gen = torch.Generator().manual_seed(seed + 31337)
    pace = ((torch.rand(N, generator=gen) < 0.02).float() * I_PACE).to(dev)
    wmot = (torch.randn(8, N, generator=torch.Generator().manual_seed(seed + 4242)) / N**0.5).to(
        dev
    )
    psi_len = 1 << N_Q
    idx = torch.arange(psi_len, device=dev)
    bits = torch.stack([((idx >> b) & 1).float() for b in range(N_Q)], 1)
    g_psi = torch.Generator().manual_seed(seed + 99)
    psi = torch.randn(psi_len, generator=g_psi) + 1j * torch.randn(psi_len, generator=g_psi)
    psi = psi.to(dev)
    psi = psi / psi.norm()

    for t in range(hist.shape[1]):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi = float(z.mean())
        cur = w_in[hist[:, t]] + pace.unsqueeze(0) - 1.0 * (f_run / 0.08) * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        nz = torch.where(ref > 0, torch.zeros_like(nv), ((nv - 0.5) / 0.5 > 0).float())
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        f_run = 0.9 * f_run + 0.1 * phi

    syms, rates = [], []
    sym = hist[:, -1]
    for _t in range(t_closed):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        phi = float(z.mean())
        cur = w_in[sym] + pace.unsqueeze(0) - 1.0 * phi
        nv = 0.95 * v + cur + rec - z * 0.5
        nz = torch.where(ref > 0, torch.zeros_like(nv), ((nv - 0.5) / 0.5 > 0).float())
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        rates.append(float(z.mean()))
        if use_quantum:
            g = z.mean(0).reshape(GROUPS, N // GROUPS).mean(1)
            theta = omega * g[:N_Q]  # 只取前 N_Q 组喂给 N_Q 个量子比特
            psi = psi * torch.exp(1j * (bits @ theta))
            psi = hadamard_layer(psi, N_Q)
            psi = cnot_ladder(psi, N_Q)
            psi = hadamard_layer(psi, N_Q)
            psi = psi / psi.norm()
            p = (psi.abs() ** 2).clamp(min=0).cpu()
            p = p / p.sum().clamp(min=1e-12)
            # **生成器在 CPU、张量也必须在 CPU**（否则 CUDA assert）
            s = int(torch.multinomial(p, 1, generator=torch.Generator().manual_seed(true_seed())))
            sym = torch.tensor([s & 7], device=dev)
            syms.append(s & 7)
        else:
            # **必须在 8 个运动输出上取 argmax**（8xN 的投影），
            # 写成 (wmot * z).sum(0).argmax() 会在 N=1024 上取，w_in 只有 64 行 ⇒ 越界
            s = int((wmot @ z.mean(0)).argmax())
            sym = torch.tensor([s], device=dev)
            syms.append(s)
    return syms, rates


def entropy_norm(syms) -> float:
    n = len(syms)
    cnt: dict = {}
    for s in syms:
        cnt[s] = cnt.get(s, 0) + 1
    return -sum((c / n) * math.log(c / n) for c in cnt.values()) / math.log(8)


def rate_dist(a, b) -> float:
    x, y = torch.tensor(a), torch.tensor(b)
    return float((x - y).norm() / (0.5 * (x.norm() + y.norm()) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--t-closed", type=int, default=T_CLOSED)
    ap.add_argument("--omega", type=float, default=8.0)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {"cells": {}}
    for mode in ("classical", "quantum"):
        rows = []
        for sd in range(args.seeds):
            net = build_pools(sd, dev, K, 0.0)
            ha = make_family_hist(0, sd * 97 + 1, T_HIST, 1, dev)
            ha2 = make_family_hist(0, sd * 97 + 11, T_HIST, 1, dev)
            hb = make_family_hist(5, sd * 97 + 2, T_HIST, 1, dev)
            sa, ra = run_loop(net, sd, ha, dev, args.t_closed, mode == "quantum", args.omega)
            _s2, ra2 = run_loop(net, sd, ha2, dev, args.t_closed, mode == "quantum", args.omega)
            _sb, rb = run_loop(net, sd, hb, dev, args.t_closed, mode == "quantum", args.omega)
            inwin = sum(1 for r in ra if 0.01 <= r <= 0.20) / len(ra)
            dh, dn = rate_dist(ra, rb), rate_dist(ra, ra2)
            rows.append(
                {
                    "seed": sd,
                    "in_window": inwin,
                    "d_hist": dh,
                    "d_null": dn,
                    "ratio": dh / max(dn, 1e-12),
                    "entropy": entropy_norm(sa),
                }
            )
            print(
                f"  {mode:9s} seed {sd}: 窗内 {inwin:.2f}  熵 {entropy_norm(sa):.2f}  "
                f"历史 {dh:.3f} / 对照 {dn:.3f} = {dh / max(dn, 1e-12):.2f}x",
                flush=True,
            )
        m = lambda k, _r=rows: sum(r[k] for r in _r) / len(_r)  # noqa: E731
        v = {k: m(k) for k in rows[0]}
        v["rows"] = rows
        res["cells"][mode] = v
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1, default=float),
            encoding="utf-8",
            newline=chr(10),
        )

    print("")
    print("判：")
    cc = res["cells"]
    q, c = cc["quantum"], cc["classical"]
    ok0 = q["in_window"] >= 0.8 and c["in_window"] >= 0.8
    print(
        f"  AY-0 两版都不熄火：量子 {q['in_window']:.2f} / 经典 {c['in_window']:.2f} "
        f"=> {'过' if ok0 else '不过 => 作废'}"
    )
    if not ok0:
        return 0
    print(
        f"  AY-1 量子版带历史 >1.5：{q['ratio']:.2f}x（经典版 {c['ratio']:.2f}x）"
        f" => {'过' if q['ratio'] > 1.5 else '不过'}"
    )
    print(
        f"  AY-2 行为非平凡：量子熵 {q['entropy']:.2f}、经典熵 {c['entropy']:.2f} "
        f"=> {'过' if q['entropy'] > 0.5 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

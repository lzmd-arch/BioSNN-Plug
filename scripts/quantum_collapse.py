"""波函数坍缩作为混沌来源：弱测量强度 η 的扫描。

预注册：`research/sel/prereg-quantum-collapse-v1.md`

## 定理（本轮的依据）

酉演化是**等距**的：`‖Uψ₁−Uψ₂‖ = ‖ψ₁−ψ₂‖` ⇒ 最大 Lyapunov 指数恒为 0
⇒ **纯酉动力学不可能混沌**。这解释了 v12 的 `K≈0.015`。

坍缩是**非线性 + 随机**的，正是缺的那一块。

## 输出只取一个标量

`φ(t) = Re⟨ψ|H|ψ⟩` —— 观测者能看到的就是这个数。**不读 ψ 的内部结构。**
"""

from __future__ import annotations

# ruff: noqa: N803, N806
import argparse
import json
import math
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.chaos_probe import k01, lyap, verdict
from scripts.quantum_phase import build_J, make_U

OUT = _pathlib.Path("sweep_results/quantum_collapse")


def series(eta: float, Gamma: float, g: float, gamma: float, steps: int, seed: int = 0):
    """弱测量强度 `eta` 下，输出标量序列 `⟨ψ|H|ψ⟩`。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    J = build_J(4.0, 32, 16.0, 0, dev)
    U = make_U(J)
    H = J + Gamma * torch.eye(J.shape[0], device=dev, dtype=J.dtype)
    gen = torch.Generator().manual_seed(seed)
    psi = (torch.randn(J.shape[0], generator=gen) + 1j * torch.randn(J.shape[0], generator=gen)).to(
        dev
    )
    psi = psi / psi.norm()
    lin = U * complex(math.cos(-0.05 * Gamma), math.sin(-0.05 * Gamma))
    dt = 0.05
    out = []
    for _ in range(steps):
        psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
        psi = lin @ psi
        psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
        psi = psi * math.exp(-dt * gamma)
        if eta > 0:
            prob = psi.abs() ** 2
            k = int(torch.multinomial(prob / prob.sum(), 1)[0])
            ek = torch.zeros_like(psi)
            ek[k] = 1.0
            psi = (1.0 - eta) * psi + eta * psi[k] * ek
            psi = psi / psi.norm()
        # **必须按模长归一**：`ψ` 在衰减项下模长指数缩小，
        # 不归一的 "⟨H⟩" 混进了模长衰减，不是能量（v13 第一次跑就栽在这）
        nrm = float((psi.conj() * psi).sum().real)
        out.append(float((psi.conj() * (H @ psi)).sum().real) / max(nrm, 1e-30))
    return np.array(out)


def divergence(eta: float, Gamma: float, g: float, gamma: float, steps: int, eps: float = 1e-9):
    """**直接检验敏感依赖**：两条只差 `eps` 的初态，看输出差如何增长。

    这本来就是混沌的定义，不依赖任何统计量。
    输出仍只取标量 `⟨H⟩`，看的是 `|φ_a(t) − φ_b(t)|`。

    纯酉演化是等距的 ⇒ 这个差**不可能**指数增长（定理）。
    若它增长了，那只能是坍缩带来的非线性/随机。
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    J = build_J(4.0, 32, 16.0, 0, dev)
    U = make_U(J)
    H = J + Gamma * torch.eye(J.shape[0], device=dev, dtype=J.dtype)
    gen = torch.Generator().manual_seed(0)
    a = (torch.randn(J.shape[0], generator=gen) + 1j * torch.randn(J.shape[0], generator=gen)).to(
        dev
    )
    a = a / a.norm()
    b = a + eps * torch.randn(J.shape[0], generator=gen).to(dev) / math.sqrt(J.shape[0])
    b = b / b.norm()
    lin = U * complex(math.cos(-0.05 * Gamma), math.sin(-0.05 * Gamma))
    dt = 0.05
    diffs = []
    for _ in range(steps):
        for name in ("a", "b"):
            psi = a if name == "a" else b
            psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
            psi = lin @ psi
            psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
            psi = psi * math.exp(-dt * gamma)
            if eta > 0:
                prob = psi.abs() ** 2
                k = int(torch.multinomial(prob / prob.sum(), 1)[0])
                ek = torch.zeros_like(psi)
                ek[k] = 1.0
                psi = (1.0 - eta) * psi + eta * psi[k] * ek
                psi = psi / psi.norm()
            if name == "a":
                a = psi
                na = float((psi.conj() * psi).sum().real)
                pa = float((psi.conj() * (H @ psi)).sum().real) / max(na, 1e-30)
            else:
                b = psi
                nb = float((psi.conj() * psi).sum().real)
                pb = float((psi.conj() * (H @ psi)).sum().real) / max(nb, 1e-30)
        diffs.append(abs(pa - pb))
    return np.array(diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--Gamma", type=float, default=0.4)
    ap.add_argument("--etas", default="0,0.05,0.2,0.5,1.0")
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    res: dict = {"Gamma": args.Gamma, "cells": {}}

    print(f"敏感依赖检验（Γ={args.Gamma}）：", flush=True)
    div_res = {}
    for e in [float(x) for x in args.etas.split(",")]:
        d = divergence(e, args.Gamma, 1.0, 0.05, 600)
        d0 = max(d[0], 1e-30)
        growth = float(d[-1] / d0)
        div_res[f"{e:.2f}"] = {"growth": growth, "d0": float(d[0]), "d_end": float(d[-1])}
        print(
            f"  η={e:4.2f}  d(0)={d[0]:.2e}  d(end)={d[-1]:.2e}  增长 {growth:.2e} 倍", flush=True
        )
    res["divergence"] = div_res
    print("", flush=True)
    print(f"弱测量强度扫描（Γ={args.Gamma}）：", flush=True)
    for e in [float(x) for x in args.etas.split(",")]:
        s = series(e, args.Gamma, 1.0, 0.05, args.steps, 0)
        if not np.all(np.isfinite(s)) or s.std() < 1e-6:
            res["cells"][f"{e:.2f}"] = {"note": "退化或非有限", "std": float(np.std(s))}
            print(f"  η={e:4.2f}  ✗ 序列退化 std={np.std(s):.2e}", flush=True)
            continue
        K, L = k01(s), lyap(s)
        res["cells"][f"{e:.2f}"] = {
            "K": K,
            "lambda": L,
            "verdict": verdict(K, L),
            "std": float(s.std()),
        }
        print(
            f"  η={e:4.2f}  K={K:+.3f}  λ={L:+.4f}  std={s.std():.3e}  => {verdict(K, L)}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )

    print("")
    print("判：")
    cc = {k: v for k, v in res["cells"].items() if "K" in v}
    if not cc:
        print("  ✗ 全部退化")
        return 1
    ks = {k: v["K"] for k, v in cc.items()}
    c1 = any(v > 0.7 for v in ks.values())
    print(
        f"  C-1 坍缩带来混沌（存在 K>0.7）：{ {k: round(v, 3) for k, v in ks.items()} } => "
        f"{'过' if c1 else '不过'}"
    )
    order = sorted(cc, key=float)
    seq = [cc[k]["K"] for k in order]
    c2 = all(seq[i] <= seq[i + 1] + 0.1 for i in range(len(seq) - 1))
    print(
        f"  C-2 剂量–反应：K 随 η = {[round(x, 3) for x in seq]} 单调不降 => {'过' if c2 else '不过'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

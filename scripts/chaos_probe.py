"""黑箱混沌判据：只吃输出标量序列，不读任何内部状态。

预注册：`research/sel/prereg-blackbox-chaos-v1.md`

## 用户的纪律

> 「不看过程只检查结果，防止混沌系统人为不混沌。」

**物理上更正**：波函数坍缩在经典仿真里不存在。
**方法论上完全成立**：今晚四次无效比较都是"测量装置改变了被测对象"。

所以本模块**只接受一个标量序列**，别的什么都不吃。

## 两个判据

| 判据 | 来源 | 混沌 | 规则 |
| :--- | :--- | :--- | :--- |
| `K` | 0–1 test (Gottwald & Melbourne) | ≈1 | ≈0 |
| `λ` | 最大 Lyapunov 指数（Rosenstein） | >0 | ≤0 |

**X-1 先自检**：正弦与周期 3 必须给 `K<0.3`，Logistic r=4 必须给 `K>0.7`。
自检不过，后面的判定全部作废。
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

OUT = _pathlib.Path("sweep_results/blackbox_chaos")


def k01(phi, n_c: int = 20, seed: int = 0) -> float:
    """0–1 test 的 K 统计量。`K≈1` 混沌，`K≈0` 规则。"""
    phi = np.asarray(phi, dtype=float)
    N = phi.size
    n = np.arange(1, N + 1)
    g = np.random.default_rng(seed)
    cs = g.uniform(0.5, 3.0, n_c)
    ks = []
    nmax = max(2, N // 10)
    ns = np.arange(1, nmax)
    for c in cs:
        p = np.cumsum(phi * np.cos(n * c))
        q = np.cumsum(phi * np.sin(n * c))
        M = np.array([np.mean((p[s:] - p[:-s]) ** 2 + (q[s:] - q[:-s]) ** 2) for s in ns])
        if M.std() < 1e-12:
            ks.append(0.0)
            continue
        ks.append(float(np.corrcoef(ns, M)[0, 1]))
    return float(np.median(ks))


def _embed(x, m: int, tau: int):
    N = x.size - (m - 1) * tau
    return np.stack([x[i * tau : i * tau + N] for i in range(m)], axis=1)


def lyap(phi, m: int = 3, tau: int = 1, horizon: int = 20, exclude: int = 10) -> float:
    """Rosenstein 法：从标量序列估计最大 Lyapunov 指数。"""
    x = np.asarray(phi, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    X = _embed(x, m, tau)
    N = X.shape[0]
    if 2 * horizon + 2 > N:
        return float("nan")
    # 只在前半段找最近邻，避免用未来信息
    idx = np.arange(0, N - horizon)
    d = np.linalg.norm(X[idx][:, None, :] - X[None, :, :], axis=2)
    for i, ii in enumerate(idx):
        lo, hi = max(0, ii - exclude), min(N, ii + exclude + 1)
        d[i, lo:hi] = np.inf
    nn = d.argmin(1)
    div = np.zeros(horizon)
    for k in range(horizon):
        ok = (idx + k < N) & (nn + k < N)
        if not ok.any():
            break
        dd = np.linalg.norm(X[idx[ok] + k] - X[nn[ok] + k], axis=1)
        div[k] = np.mean(dd[dd > 0]) if (dd > 0).any() else 1e-12
    div = np.maximum(div, 1e-12)
    ks = np.arange(1, horizon)
    slope = np.polyfit(ks, np.log(div[1:horizon]), 1)[0]
    return float(slope)


def verdict(K: float, lam: float):
    ck = K > 0.7
    cl = lam > 0.0
    if ck == cl:
        return "混沌" if ck else "规则"
    return "未定（两判据不一致）"


def self_check() -> dict:
    """X-1：已知序列必须给出已知答案。"""
    n = np.arange(4000)
    sine = np.sin(0.05 * n)
    per3 = np.array([1.0, 2.0, 3.0] * 1334)[:4000]
    x = 0.4
    logi = np.empty(4000)
    for i in range(4000):
        x = 4.0 * x * (1.0 - x)
        logi[i] = x
    out = {}
    for name, s in (("正弦", sine), ("周期3", per3), ("Logistic r=4", logi)):
        K = k01(s)
        L = lyap(s)
        out[name] = {"K": K, "lambda": L, "verdict": verdict(K, L)}
        print(f"  {name:14s} K={K:+.3f}  λ={L:+.4f}  => {out[name]['verdict']}", flush=True)
    ok = out["正弦"]["K"] < 0.3 and out["周期3"]["K"] < 0.3 and out["Logistic r=4"]["K"] > 0.7
    return {"cases": out, "ok": ok}


def series_quantum(Gamma: float, g: float, gamma: float, J_scale: float, steps: int, seed: int):
    """复数哈密顿系统的**标量输出序列**：每步 mean|ψ|²。"""
    import torch

    from scripts.quantum_phase import build_J, make_U

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    J = build_J(4.0, 32, 16.0, 0, dev) * J_scale
    U = make_U(J)
    gen = torch.Generator().manual_seed(seed)
    psi = torch.randn(J.shape[0], generator=gen) + 1j * torch.randn(J.shape[0], generator=gen)
    psi = psi.to(dev)
    psi = psi / psi.norm()
    lin = U * complex(math.cos(-0.05 * Gamma), math.sin(-0.05 * Gamma))
    out = []
    dt = 0.05
    for _ in range(steps):
        psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
        psi = lin @ psi
        psi = psi * torch.exp(-0.5j * dt * g * psi.abs() ** 2)
        psi = psi * math.exp(-dt * gamma)
        out.append(float((psi.abs() ** 2).mean()))
    return np.array(out)


def series_lif(steps: int, seed: int, scale: float = 1.86):
    """我们的 LIF 的**标量输出序列**：每步 mean z。"""
    import torch

    from scripts.bio_all import build
    from scripts.cortical_fold import N

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_in, pre, wrec, _a, _d, _p = build(N, set(), seed, dev, 8)
    toks = torch.randint(0, 64, (32, steps), generator=torch.Generator().manual_seed(seed)).to(dev)
    v = torch.zeros((32, N), device=dev)
    z = torch.zeros_like(v)
    ref = torch.zeros_like(v)
    out = []
    for t in range(steps):
        rec = torch.zeros_like(z)
        for j in range(pre.shape[1]):
            rec = rec + z[:, pre[:, j]] * wrec[:, j].unsqueeze(0)
        nv = 0.95 * v + w_in[toks[:, t]] * scale + rec - z * 0.5
        fresh = ((nv - 0.5) / 0.5 > 0).float()
        nz = torch.where(ref > 0, torch.zeros_like(fresh), fresh)
        ref = torch.clamp(ref + 2.0 * nz - 1.0, 0.0, 2.0)
        v, z = nv, nz
        out.append(float(z.mean()))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", type=_pathlib.Path, default=OUT)
    args = ap.parse_args()
    res: dict = {}

    print("X-1 判据自检：", flush=True)
    sc = self_check()
    res["self_check"] = sc
    print(
        f"  => {'过（工具可用）' if sc['ok'] else '**不过 ⇒ 工具坏了，后面全部作废**'}", flush=True
    )
    if not sc["ok"]:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
        )
        return 1

    print("", flush=True)
    print("复数哈密顿系统（不同 Γ）：", flush=True)
    qres = {}
    for Gm in (0.1, 0.4, 1.6):
        s = series_quantum(Gm, 1.0, 0.05, 1.0, args.steps, 0)
        if not np.all(np.isfinite(s)):
            qres[f"{Gm}"] = {"note": "非有限"}
            print(f"  Γ={Gm:4.1f}  序列含非有限值", flush=True)
            continue
        K, L = k01(s), lyap(s)
        qres[f"{Gm}"] = {"K": K, "lambda": L, "verdict": verdict(K, L), "std": float(s.std())}
        print(
            f"  Γ={Gm:4.1f}  K={K:+.3f}  λ={L:+.4f}  std={s.std():.3e}  => {verdict(K, L)}",
            flush=True,
        )
    res["quantum"] = qres

    print("", flush=True)
    print("我们的 LIF（参照）：", flush=True)
    s = series_lif(args.steps, 0)
    K, L = k01(s), lyap(s)
    res["lif"] = {"K": K, "lambda": L, "verdict": verdict(K, L), "std": float(s.std())}
    print(f"  LIF  K={K:+.3f}  λ={L:+.4f}  std={s.std():.3e}  => {verdict(K, L)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8", newline=chr(10)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

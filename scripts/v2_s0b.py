"""S0b：标定型（乘性）算子。

`z_c = log q(c)/τ(p) + (p·W)·A_c`，`τ(p) = softplus(p·w + b0) + 0.1`

复用 S0 落盘的同一份 raw.json（同一批位置、同一划分），不重新打模型。
预注册：`research/sel/prereg-v2-s0b-multiplicative.md`
"""

from __future__ import annotations

# ruff: noqa: N803, N806  —— 矩阵记号
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.v2_s0 import OUT, _proj, _setup

#: tau = softplus(s) + 0.1 == 1 的那个 s。**恒等温度必须是默认值**——
#: 第一版忘了这一条，B0 拿到的其实是「被锐化过的 Qwen」（tau=0.79），
#: 于是 B3 的增益被虚报成 +1.80 nats。
S_TAU1 = math.log(math.expm1(0.9))


def _masked_lp(z, mask):
    if mask is not None:
        z = np.where(mask, z, -1e9)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def _tau(P, w, b0):
    s = P @ w + b0
    return np.logaddexp(0.0, s) + 0.1


def _forward(logq, P, w, b0, A, mask):
    tau = _tau(P, w, b0)[:, None]
    z = logq / tau
    if A is not None:
        r = A.shape[0]
        z = z + (P @ _proj(r)) @ A
    return z, tau, _masked_lp(z, mask)


def fit(logq, y, P, mask, r=0, l2=1e-2, epochs=150, lr=0.05):
    n, C = logq.shape
    w = np.zeros(8)
    b0 = np.array(S_TAU1)  # 从 tau=1（恒等）起步
    A = np.zeros((r, C)) if r > 0 else None
    Q = P @ _proj(r) if r > 0 else None
    for _ in range(epochs):
        _, tau, lp = _forward(logq, P, w, b0, A, mask)
        g = np.exp(lp)
        g[np.arange(n), y] -= 1.0
        g /= n
        dz = np.where(mask, g, 0.0)
        if A is not None:
            A = A - lr * (Q.T @ dz + l2 * A)
        # τ 的梯度：z = logq/τ，∂z/∂τ = −logq/τ²
        dtau = (dz * (-logq / tau**2)).sum(1)
        ds = dtau * (1.0 / (1.0 + np.exp(-(P @ w + b0))))
        w = w - lr * (P.T @ ds / 1.0)
        b0 = b0 - lr * ds.sum()
    return w, b0, A


def score(logq, y, P, mask, w=None, b0=S_TAU1, A=None):
    lp = _forward(logq, P, w if w is not None else np.zeros(8), b0, A, mask)[2]
    return float(lp[np.arange(len(y)), y].mean())


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    raw = json.loads((OUT / (f"raw_{tag}.json" if tag else "raw.json")).read_text(encoding="utf-8"))
    rows = raw["rows"]
    logq, y, P, _, cover, mask = _setup(rows, 20)
    n = len(rows)
    order = np.random.default_rng(12345).permutation(n)
    cut = int(n * 0.7)
    tr, te = order[:cut], order[cut:]
    mtr, mte = mask[tr], mask[te]
    out = {"positions": n, "truth_coverage": round(cover, 4), "train": len(tr), "holdout": len(te)}

    out["B0"] = score(logq[te], y[te], P[te], mte)
    best_t, best = 1.0, -1e9
    for t in np.linspace(0.5, 6.0, 111):
        s = score(logq[tr], y[tr], P[tr], mtr, b0=math.log(math.exp(t - 0.1) - 1.0))
        if s > best:
            best, best_t = s, float(t)
    out["B2"] = {
        "logp": score(logq[te], y[te], P[te], mte, b0=math.log(math.exp(best_t - 0.1) - 1.0)),
        "temp": best_t,
    }

    print("  B3 拟合", flush=True)
    w3, b3, _ = fit(logq[tr], y[tr], P[tr], mtr, r=0)
    out["B3"] = {
        "logp": score(logq[te], y[te], P[te], mte, w=w3, b0=b3),
        "gain_vs_B0": score(logq[te], y[te], P[te], mte, w=w3, b0=b3) - out["B0"],
        "mean_tau": float(_tau(P[te], w3, b3).mean()),
    }

    print("  B4 拟合", flush=True)
    w4, b4, A4 = fit(logq[tr], y[tr], P[tr], mtr, r=2)
    out["B4"] = {
        "logp": score(logq[te], y[te], P[te], mte, w=w4, b0=b4, A=A4),
        "gain_vs_B3": score(logq[te], y[te], P[te], mte, w=w4, b0=b4, A=A4) - out["B3"]["logp"],
    }

    perm = []
    print("  置换检验：100 次", flush=True)
    for i in range(100):
        pi = np.random.default_rng(i).permutation(len(tr))
        wi, bi, _ = fit(logq[tr], y[tr], P[tr][pi], mtr, r=0, epochs=60)
        perm.append(score(logq[te], y[te], P[te], mte, w=wi, b0=bi) - out["B0"])
        if i % 25 == 0:
            print(f"    {i}/100", flush=True)
    out["S0b-3_p"] = float((np.array(perm) >= out["B3"]["gain_vs_B0"]).mean())
    out["perm_mean"] = float(np.mean(perm))

    (OUT / (f"s0b_{tag}.json" if tag else "s0b.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

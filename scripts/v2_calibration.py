"""千问在真实文本上的**标定**报告：全局温度 vs 自适应 τ(p)。

用 S0 落盘的 540 个位置（Qwen3-4B-Q4_K_M，本地 llama-server），不再打模型。

三件事：
1. 全局温度扫描 —— 真值平均 log p、ECE、风险–覆盖曲线；
2. 自适应 τ(p)（9 个参数）在同一份数据上的表现；
3. **明确指出**：温度**不改变 argmax**，所以它不会提高 top-1，只影响置信度类决策。

预注册：`research/sel/prereg-v2-s0b-multiplicative.md`
"""

from __future__ import annotations

# ruff: noqa: N803, N806

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.v2_s0 import OUT, _setup  # noqa: E402
from scripts.v2_s0b import S_TAU1, _masked_lp, fit  # noqa: E402


def _ece(prob: np.ndarray, hit: np.ndarray, bins: int = 10) -> float:
    """期望标定误差：把"自报概率"分箱，和实际命中率比。"""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        sel = (prob >= edges[i]) & (prob < edges[i + 1])
        if sel.sum() == 0:
            continue
        e += sel.mean() * abs(prob[sel].mean() - hit[sel].mean())
    return float(e)


def _report(logq, y, mask, name: str) -> dict:
    lp = _masked_lp(logq, mask)
    n = len(y)
    prob_true = np.exp(lp[np.arange(n), y])
    pred = lp.argmax(1)
    hit = (pred == y).astype(float)
    conf = np.exp(lp.max(1))
    order = np.argsort(-conf)
    half = order[: n // 2]
    return {
        "arm": name,
        "mean_logp": float(lp[np.arange(n), y].mean()),
        "top1": float(hit.mean()),
        "ece": _ece(conf, hit),
        "acc@50%coverage": float(hit[half].mean()),
    }


def main() -> int:
    rows = json.loads((OUT / "raw.json").read_text(encoding="utf-8"))["rows"]
    logq, y, P, _, cover, mask = _setup(rows, 20)
    n = len(rows)
    order = np.random.default_rng(12345).permutation(n)
    cut = int(n * 0.7)
    tr, te = order[:cut], order[cut:]
    mtr, mte = mask[tr], mask[te]

    out = {"positions": n, "train": len(tr), "holdout": len(te), "truth_coverage": round(cover, 4)}

    grid = []
    for t in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.65, 4.5, 6.0, 8.0, 10.0, 12.0]:
        b0 = math.log(math.expm1(t - 0.1))
        r = _report(logq[te] / t, y[te], mte, f"T={t:g}")
        r["T"] = t
        grid.append(r)
    out["temperature_grid"] = grid
    best = max(grid, key=lambda r: r["mean_logp"])
    out["best_T"] = best["T"]

    # 自适应 τ(p)：在训练折上拟合，留出折上评
    w, b0, _ = fit(logq[tr], y[tr], P[tr], mtr, r=0, epochs=150)
    s_te = P[te] @ w + b0
    tau_te = np.logaddexp(0.0, s_te) + 0.1
    r = _report(logq[te] / tau_te[:, None], y[te], mte, "tau(p)")
    r["mean_tau"] = float(tau_te.mean())
    out["adaptive"] = r

    base = _report(logq[te], y[te], mte, "T=1")
    out["base"] = base
    (OUT / "calibration.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"{'arm':>8s} {'log p':>8s} {'top1':>6s} {'ECE':>7s} {'acc@50%':>8s}")
    for r in [base, *grid, out["adaptive"]]:
        print(
            f"{r['arm']:>8s} {r['mean_logp']:8.4f} {r['top1']:6.3f} "
            f"{r['ece']:7.4f} {r['acc@50%coverage']:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""把四维残差的逻辑回归权重拟合出来，**固化进 runtime/semprea/verify.py**。

理由：运行时里的门现在用的是 R1 阈值规则，而实测它在同记忆负例上 AUC 只有 0.544
（接近随机），四维残差是 0.699。训练好的头不接上去，等于发了一个不起作用的门。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grounding_compare import build, memories, r1, r4, r5


def main() -> int:
    rng = random.Random(0)
    texts = memories()
    claim, evid, y, _ts, _cross = build(rng, texts)
    x = np.asarray(
        [[r1(c, e), r4(c, e), r5(c, e), len(e) / 500.0] for c, e in zip(claim, evid, strict=True)],
        dtype=np.float64,
    )
    idx = list(range(len(y)))
    rng.shuffle(idx)
    tr = idx[: int(len(idx) * 0.7)]

    mean, std = x[tr].mean(0), x[tr].std(0)
    std[std < 1e-9] = 1.0
    z = np.hstack([(x[tr] - mean) / std, np.ones((len(tr), 1))])
    w = np.zeros(z.shape[1])
    for _ in range(4000):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        w -= 0.3 * (z.T @ (p - y[tr]) / len(tr) + np.concatenate([w[:-1], [0.0]]) / len(tr))

    out = {
        "features": ["r1", "r4", "r5", "len_over_500"],
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "weights": [round(float(v), 6) for v in w],
        "samples": len(y),
        "note": "在 280 个同记忆负例样本上拟合；跨条负例不参与（会泄漏时间戳）。",
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    path = Path("sweep_results/hallucination/verify_weights.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

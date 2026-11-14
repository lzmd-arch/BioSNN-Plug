"""SNN 的**逐位置序列**能不能比池化均值更会判"有据/无据"。

前三次负结果用的都是"把序列压成一个点"（`/encode` 的均值）。这次用 `/features`
拿到逐位置特征（已投影到 64 维），取 **mean / std / max** 三种序列统计，再拼四维残差。

判据：留出 AUC 必须 **> 0.748**（四维残差的当前最好成绩），否则序列也没有信息。
标签置换检验必须通过。
"""

from __future__ import annotations

import json
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from grounding_compare import auc, build, fit, memories, r1, r4, r5

FEAT = "http://127.0.0.1:18763/features"


def seq_stats(text: str) -> np.ndarray:
    req = urllib.request.Request(
        FEAT, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"}
    )
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open(req, timeout=300) as handle:
            pos = np.asarray(json.loads(handle.read())["positions"], dtype=np.float64)
    except Exception:
        return np.zeros(64 * 3)
    if pos.ndim != 2 or len(pos) == 0:
        return np.zeros(4)
    # **只留 4 个可解释的统计量**：196 维配 349 样本必然过拟合（上轮置换 0.633 不过）。
    # 这四个量描述"状态序列的形状"：有多大、抖不抖、峰在哪、是不是单调上扬。
    norms = np.linalg.norm(pos, axis=1)
    t = np.arange(len(norms), dtype=np.float64)
    slope = float(np.polyfit(t, norms, 1)[0]) if len(norms) > 1 else 0.0
    return np.asarray([norms.mean(), norms.std(), norms.max(), slope])


def main() -> int:
    rng = random.Random(0)
    texts = memories()
    claim, evid, y, _ts, _cross = build(rng, texts)
    print(f"样本 {len(y)}", flush=True)
    seq = np.stack([seq_stats(f"{e}\n{c}") for e, c in zip(evid, claim, strict=True)])
    res = np.asarray(
        [[r1(c, e), r4(c, e), r5(c, e), len(e) / 500.0] for c, e in zip(claim, evid, strict=True)]
    )
    seq = seq / (seq.std(0) + 1e-9)

    idx = list(range(len(y)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    def run(name, x):
        m = fit(x[tr], y[tr])
        a_tr, a_te = auc(m(x[tr]), y[tr]), auc(m(x[te]), y[te])
        print(f"  {name:12s} 训练 {a_tr:.4f}  留出 {a_te:.4f}", flush=True)
        return a_te

    a_res = run("四维残差", res)
    a_seq = run("序列统计", seq)
    a_both = run("残差+序列", np.hstack([res, seq]))

    perm = np.random.default_rng(0).permutation(y)
    mperm = fit(np.hstack([res, seq])[tr], perm[tr])
    p_auc = auc(mperm(np.hstack([res, seq])[te]), perm[te])
    print(f"  标签置换 {p_auc:.4f}", flush=True)

    out = {
        "samples": len(y),
        "auc": {
            "residuals": round(a_res, 4),
            "sequence": round(a_seq, 4),
            "both": round(a_both, 4),
        },
        "perm_auc": round(float(p_auc), 4),
        "verdict": "序列有信息" if a_both > a_res + 0.02 else "序列没有额外信息（≤ 四维残差）",
    }
    print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
    p = Path("sweep_results/hallucination/snn_seq_head.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

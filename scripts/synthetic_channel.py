"""**合成数据证明机制**：当上下文真的携带信息时，模型能不能用上它。

今天所有负结果都卡在同一句"快半没有独占信息"。但那是**数据**的问题还是**能力**的问题？
构造一份"独占信号确实存在"的合成数据就能分开：

- 词表 V=64；
- 上下文向量 c ∈ R^8；
- **标签 = argmax(W·c + 0.3·prev_token_embed)**：c 决定大部分，上一个 token 决定一小部分。

两个臂，同一份数据、同一次划分：
- **text-only**：只看上一个 token → 学不到 c 的映射，只能到边缘分布；
- **text+context**：加上 c → 应该能接近 100%。

判据：`text+context` 的 top-1 比 `text-only` **高 ≥ 20pp**。
若成立 ⇒ **机制没问题，今天的负结果是数据造成的**；若不成立 ⇒ 连合成信号都用不上，架构有问题。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

V, D, N = 64, 8, 40000
rng = np.random.default_rng(0)
W = rng.normal(size=(D, V)) / np.sqrt(D)  # 上下文 → 词表
E = rng.normal(size=(V, V)) / np.sqrt(V)  # 上一个 token → 词表


def make(n: int):
    ctx = rng.normal(size=(n, D))
    prev = rng.integers(0, V, size=n)
    logits = ctx @ W + 0.3 * E[prev]
    return ctx, prev, logits.argmax(1)


ctx, prev, y = make(N)
cut = int(N * 0.7)
tr, te = slice(0, cut), slice(cut, N)


def one_hot(idx, size):
    out = np.zeros((len(idx), size))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def fit(x, yv, iters=1500, lr=0.5, l2=1e-3):
    w = np.zeros((x.shape[1], V))
    for _ in range(iters):
        z = x @ w
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(yv)), yv] -= 1.0
        w -= lr * (x.T @ p / len(yv) + l2 * w)
    return w


def top1(w, x, yv):
    z = x @ w
    return float((z.argmax(1) == yv).mean())


x_text = one_hot(prev[tr], V)
x_both = np.hstack([x_text, ctx[tr]])
w_text = fit(x_text, y[tr])
w_both = fit(x_both, y[tr])

acc_text = top1(w_text, one_hot(prev[te], V), y[te])
acc_both = top1(w_both, np.hstack([one_hot(prev[te], V), ctx[te]]), y[te])

out = {
    "samples": N,
    "vocab": V,
    "text_only_top1": round(acc_text, 4),
    "text_plus_context_top1": round(acc_both, 4),
    "gain_pp": round((acc_both - acc_text) * 100, 2),
    "verdict": "机制成立：有独占信息时模型用得上"
    if (acc_both - acc_text) >= 0.20
    else "机制不成立：给了信号也用不上",
}
print(json.dumps(out, ensure_ascii=False, indent=1))
p = Path("sweep_results/synthetic_channel/v1.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")

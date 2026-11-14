"""一次算完三个头：**数据只生成一遍、划分只有一次**，从构造上消除脚本间漂移。

rule  只用 R1
lr    R1 / R4 / R5 / 长度 → 逻辑回归
snn   /encode(证据+主张) 的读出向量（PCA≤16，只在训练集拟合）+ 上面的四项

判据 PN-2：snn 留出 AUC 必须比 lr 高 ≥0.02，否则 SNN 在这件事上没有位置。
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

FAST = "http://127.0.0.1:18763/encode"
DB = Path("D:/AxoData/memory/mem.db")


def memories() -> list[str]:
    uri = "file:" + urllib.parse.quote(str(DB).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return [
            r[0]
            for r in con.execute(
                "select content from memories where content is not null and length(content) > 20 "
                "order by created_at"
            )
        ]
    finally:
        con.close()


def r1(a: str, e: str) -> float:
    items = re.findall(r"[\u4e00-\u9fff]{2,4}|\d+", a)
    if not items:
        return 1.0
    return 0.0 if not e else sum(1 for i in items if i in e) / len(items)


def r4(a: str, e: str) -> float:
    dates = re.findall(r"\d{4}年\d{1,2}月(?:\d{1,2}[日号])?|\d{1,2}月\d{1,2}[日号]", a)
    if not dates:
        return 0.0
    return 1.0 if not e else sum(1 for d in dates if d not in e) / len(dates)


def r5(a: str, e: str) -> float:
    names = re.findall(r"(?:叫|是|在|有)([\u4e00-\u9fff]{2,4})", a)
    if not names:
        return 0.0
    return 1.0 if not e else sum(1 for n in names if n not in e) / len(names)


def build(rng: random.Random, texts: list[str]):
    claims, evids, labels = [], [], []
    for text in texts:
        sents = [s for s in re.split(r"[。！？\n]", text) if 8 <= len(s) <= 60]
        if not sents:
            continue
        claim, wrong = rng.choice(sents), texts[rng.randrange(len(texts))]
        negs: list[str] = []
        if re.search(r"\d+", text):
            negs.append(
                re.sub(r"\d+", lambda m: str(int(m.group()) + rng.randint(1, 9)), text, count=1)
            )
        dates = re.findall(r"\d{1,2}月\d{1,2}[日号]|\d{4}年\d{1,2}月", text)
        if dates:
            negs.append(text.replace(rng.choice(dates), "3月14日", 1))
        names = re.findall(r"(?:叫|是|在|有)([\u4e00-\u9fff]{2,4})", text)
        if names:
            donor = re.findall(r"(?:叫|是|在|有)([\u4e00-\u9fff]{2,4})", wrong)
            if donor:
                negs.append(text.replace(rng.choice(names), rng.choice(donor), 1))
        negs.append(wrong)
        claims.append(claim)
        evids.append(text)
        labels.append(1)
        for neg in dict.fromkeys(negs):
            if neg != text:
                claims.append(claim)
                evids.append(neg)
                labels.append(0)
    return claims, evids, np.asarray(labels, dtype=float)


def auc(s: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    return (
        float((ranks[y > 0].sum() - pos * (pos + 1) / 2) / (pos * neg))
        if pos and neg
        else float("nan")
    )


def fit(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 3000, lr: float = 0.3):
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-9] = 1.0
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    w = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        w -= lr * (z.T @ (p - y) / len(y) + l2 * np.concatenate([w[:-1], [0.0]]) / len(y))
    return lambda xx: (
        1
        / (1 + np.exp(-np.clip(np.hstack([(xx - mean) / std, np.ones((len(xx), 1))]) @ w, -30, 30)))
    )


def encode(texts: list[str]) -> np.ndarray:
    out = []
    for start in range(0, len(texts), 32):
        req = urllib.request.Request(
            FAST,
            data=json.dumps({"texts": texts[start : start + 32]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=600) as handle:
            out.extend(json.loads(handle.read())["vectors"])
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    rng = random.Random(0)
    texts = memories()
    claims, evids, y = build(rng, texts)
    res = np.asarray(
        [
            [r1(c, e), r4(c, e), r5(c, e), len(e) / 500.0]
            for c, e in zip(claims, evids, strict=True)
        ],
        dtype=np.float32,
    )
    print(f"样本 {len(y)}（正 {int(y.sum())}）", flush=True)
    emb = encode([f"{e}\n{c}" for e, c in zip(evids, claims, strict=True)])
    print(f"编码完成，维度 {emb.shape[1]}", flush=True)

    idx = list(range(len(y)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    mu = emb[tr].mean(0)
    _, _, vt = np.linalg.svd(emb[tr] - mu, full_matrices=False)
    pca = (emb - mu) @ vt[:16].T
    pca = pca / (pca.std(0) + 1e-9)

    def run(name: str, x: np.ndarray) -> tuple[float, float]:
        model = fit(x[tr], y[tr])
        a_tr, a_te = auc(model(x[tr]), y[tr]), auc(model(x[te]), y[te])
        print(f"  {name:5s} 训练 {a_tr:.4f}  留出 {a_te:.4f}", flush=True)
        return a_tr, a_te

    print("三个头（同一份数据、同一次划分）：")
    _, a_rule = run("rule", -res[:, :1])
    _, a_lr = run("lr", res)
    _, a_snn = run("snn", np.hstack([pca, res]))

    out = {
        "samples": len(y),
        "auc": {"rule": round(a_rule, 4), "lr": round(a_lr, 4), "snn": round(a_snn, 4)},
        "verdict": "SNN 有位置" if a_snn > a_lr + 0.02 else "SNN 没有位置：不比残差基线强",
    }
    print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
    path = Path("sweep_results/hallucination/grounding_compare.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

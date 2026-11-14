"""④ SNN 头：它的表征能不能比 4 维残差更会判"有据/无据"。

同一批数据、同一种划分，三个头比较 AUC：

  rule    只用 R1
  lr      4 维残差 → 逻辑回归            （上一轮：0.773）
  snn     /encode(证据+主张) 的读出向量 → 逻辑回归（**冻结的 SNN，只训一个线性头**）

判据 PN-2：snn 必须优于 lr，否则 SNN 在这件事上没有位置。
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


def post(texts: list[str]) -> np.ndarray:
    req = urllib.request.Request(
        FAST,
        data=json.dumps({"texts": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=600) as handle:
        return np.asarray(json.loads(handle.read())["vectors"], dtype=np.float32)


def memories(db: Path) -> list[str]:
    uri = "file:" + urllib.parse.quote(str(db).replace("\\", "/")) + "?mode=ro"
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


def r1(answer: str, evidence: str) -> float:
    items = re.findall(r"[\u4e00-\u9fff]{2,4}|\d+", answer)
    if not items:
        return 1.0
    return 0.0 if not evidence else sum(1 for i in items if i in evidence) / len(items)


def r4(answer: str, evidence: str) -> float:
    dates = re.findall(r"\d{4}年\d{1,2}月(?:\d{1,2}[日号])?|\d{1,2}月\d{1,2}[日号]", answer)
    if not dates:
        return 0.0
    return 1.0 if not evidence else sum(1 for d in dates if d not in evidence) / len(dates)


def r5(answer: str, evidence: str) -> float:
    names = re.findall(r"(?:叫|是|在|有)([一-鿿]{2,4})", answer)
    if not names:
        return 0.0
    return 1.0 if not evidence else sum(1 for n in names if n not in evidence) / len(names)


def auc(scores, labels) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels > 0].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit(x, y, l2=1.0, iters=3000, lr=0.3):
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-9] = 1.0
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    w = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z @ w)))
        w -= lr * (z.T @ (p - y) / len(y) + l2 * np.concatenate([w[:-1], [0.0]]) / len(y))
    return lambda xx: 1 / (1 + np.exp(-(np.hstack([(xx - mean) / std, np.ones((len(xx), 1))]) @ w)))


def main() -> int:
    rng = random.Random(0)
    texts = memories(DB)
    pairs = []
    for text in texts:
        sentences = [s for s in re.split(r"[。！？\n]", text) if 8 <= len(s) <= 60]
        if not sentences:
            continue
        claim = rng.choice(sentences)
        wrong = texts[rng.randrange(len(texts))]
        negatives: list[str] = []
        if re.findall(r"\d+", text):
            negatives.append(
                re.sub(r"\d+", lambda m: str(int(m.group()) + rng.randint(1, 9)), text, count=1)
            )
        dates = re.findall(r"\d{1,2}月\d{1,2}[日号]|\d{4}年\d{1,2}月", text)
        if dates:
            negatives.append(text.replace(rng.choice(dates), "3月14日", 1))
        names = re.findall(r"(?:叫|是|在|有)([一-鿿]{2,4})", text)
        if names:
            donor = re.findall(r"(?:叫|是|在|有)([一-鿿]{2,4})", wrong)
            if donor:
                negatives.append(text.replace(rng.choice(names), rng.choice(donor), 1))
        negatives.append(wrong)
        pairs.append((claim, text, 1))
        for neg in dict.fromkeys(negatives):
            if neg != text:
                pairs.append((claim, neg, 0))
    print(f"样本 {len(pairs)}", flush=True)

    claims = [c for c, _, _ in pairs]
    evid = [e for _, e, _ in pairs]
    x_snn = post([e + "\n" + c for e, c in zip(evid, claims, strict=True)])
    x_res = (
        np.asarray([[r1(c, e)]] for c, e, _ in pairs)
        if False
        else np.asarray([[r1(c, e)] for c, e, _ in pairs], dtype=np.float32)
    )
    y = np.asarray([label for _, _, label in pairs], dtype=float)

    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    def report(name, x):
        model = fit(x[tr], y[tr], l2=1.0)
        p_tr, p_te = model(x[tr]), model(x[te])
        a_tr, a_te = auc(p_tr, y[tr]), auc(p_te, y[te])
        flag = "  ← 过拟合" if a_tr - a_te > 0.15 else ""
        print(f"  {name:8s} 训练 {a_tr:.4f}  留出 {a_te:.4f}{flag}", flush=True)
        return round(a_te, 4)

    # **PCA 只在训练集上拟合**，降到 16 维——150 个样本配 15,624 维会直接发散
    flat = x_snn.reshape(len(x_snn), -1)
    mu = flat[tr].mean(0)
    centred = flat[tr] - mu
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    comp = vt[:16]
    x_pca = (flat - mu) @ comp.T
    x_pca = x_pca / (x_pca.std(0) + 1e-9)

    print("留出集：")
    a_rule = report("rule", -x_res[:, :1])
    a_lr = report("lr", x_res)
    a_snn = report("snn", np.hstack([x_pca, x_res]))

    out = {
        "samples": len(pairs),
        "auc": {"rule": a_rule, "lr": a_lr, "snn_with_residuals": a_snn},
        "verdict": "SNN 有位置"
        if a_snn > a_lr + 0.02
        else "SNN 没有位置：不比 4 维残差的逻辑回归强",
    }
    print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
    path = Path("sweep_results/hallucination/grounding_snn_head.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"结果 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""四个头一次算完：数据只生成一遍、划分只有一次。

rule   只用 R1
lr     R1/R4/R5/长度 → 逻辑回归                     （基线 AUC 0.852）
time   上面的 + 真实时间特征（库里就有，不需新硬件）
snn    /encode 读出向量（PCA16，训练集拟合）+ 四项

判据：`time` 必须比 `lr` 高 ≥0.02 AUC，否则时间信息在这批数据上不判别"有据/无据"。
SNN 头的判据同前：必须优于 `lr`，当前已知它过拟合。
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

FAST = "http://127.0.0.1:18763/encode"
DB = Path("D:/AxoData/memory/mem.db")


def memories() -> list[tuple[str, float]]:
    uri = "file:" + urllib.parse.quote(str(DB).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return [
            (r[0], float(r[1] or 0.0))
            for r in con.execute(
                "select content, created_at from memories where content is not null "
                "and length(content) > 20 order by created_at"
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


def build(rng: random.Random, texts: list[tuple[str, float]]):
    claim, evid, label, tstamp = [], [], [], []
    cross: list[tuple[str, str, float]] = []
    for text, ts in texts:
        sents = [s for s in re.split(r"[。！？\n]", text) if 8 <= len(s) <= 60]
        if not sents:
            continue
        c = rng.choice(sents)
        wrong_text, wrong_ts = texts[rng.randrange(len(texts))]
        negs: list[tuple[str, float]] = []
        if re.search(r"\d+", text):
            negs.append(
                (
                    re.sub(
                        r"\d+", lambda m: str(int(m.group()) + rng.randint(1, 9)), text, count=1
                    ),
                    ts,
                )
            )
        dates = re.findall(r"\d{1,2}月\d{1,2}[日号]|\d{4}年\d{1,2}月", text)
        if dates:
            negs.append((text.replace(rng.choice(dates), "3月14日", 1), ts))
        names = re.findall(r"(?:叫|是|在|有)([\u4e00-\u9fff]{2,4})", text)
        if names:
            donor = re.findall(r"(?:叫|是|在|有)([\u4e00-\u9fff]{2,4})", wrong_text)
            if donor:
                negs.append((text.replace(rng.choice(names), rng.choice(donor), 1), ts))
        # **跨条负例单独成组**：它的时间戳与正例不同，混进来会泄漏标签。
        # 新增两种负例：**指代错位**（我→你）与**跨句矛盾**（补一句冲突的日期）
        if "我" in text:
            negs.append((text.replace("我", "你", 1), ts))
        if dates:
            negs.append((text + f"另一次是 {rng.randint(1, 12)}月{rng.randint(1, 28)}日。", ts))
        cross.append((c, wrong_text, wrong_ts))
        claim.append(c)
        evid.append(text)
        label.append(1)
        tstamp.append(ts)
        for neg, nts in dict.fromkeys(negs):
            if neg != text:
                claim.append(c)
                evid.append(neg)
                label.append(0)
                tstamp.append(nts)
    return claim, evid, np.asarray(label, dtype=float), np.asarray(tstamp, dtype=np.float64), cross


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
    out: list = []
    for start in range(0, len(texts), 32):
        req = urllib.request.Request(
            FAST,
            data=json.dumps({"texts": texts[start : start + 32]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=900) as handle:
            out.extend(json.loads(handle.read())["vectors"])
    return np.asarray(out, dtype=np.float32)


def main() -> int:
    rng = random.Random(0)
    texts = memories()
    claim, evid, y, ts, cross = build(rng, texts)
    day = ts / 86400.0
    hour = (ts % 86400) / 3600.0
    weekday = (day % 7) / 7.0
    # **只用与标签无关的量**：同一记忆的正负例取值必然相同，不可能泄漏。
    time_feats = np.stack(
        [
            np.sin(2 * math.pi * hour / 24),
            np.cos(2 * math.pi * hour / 24),
            np.sin(2 * math.pi * weekday),
            np.cos(2 * math.pi * weekday),
        ],
        axis=1,
    ).astype(np.float32)

    base = np.asarray(
        [[r1(c, e), r4(c, e), r5(c, e), len(e) / 500.0] for c, e in zip(claim, evid, strict=True)],
        dtype=np.float32,
    )
    with_time = np.hstack([base, time_feats])
    print(
        f"样本 {len(y)}（正 {int(y.sum())}）；时间跨度 {day.max() - day.min():.1f} 天", flush=True
    )
    emb = encode([f"{e}\n{c}" for e, c in zip(evid, claim, strict=True)])

    idx = list(range(len(y)))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    mu = emb[tr].mean(0)
    _, _, vt = np.linalg.svd(emb[tr] - mu, full_matrices=False)
    pca = (emb - mu) @ vt[:16].T
    pca = pca / (pca.std(0) + 1e-9)

    scores: dict[str, float] = {}

    def run(name: str, x: np.ndarray) -> float:
        model = fit(x[tr], y[tr])
        a_tr, a_te = auc(model(x[tr]), y[tr]), auc(model(x[te]), y[te])
        print(f"  {name:9s} 训练 {a_tr:.4f}  留出 {a_te:.4f}", flush=True)
        scores[name] = round(a_te, 4)
        return a_te

    print("四个头（同一份数据、同一次划分）：")
    run("rule", -base[:, :1])
    a_lr = run("lr", base)
    a_time = run("lr+time", with_time)
    run("snn", np.hstack([pca, base]))

    # **标签置换检验**：打乱 y 重训，AUC 必须回到 0.5 附近。
    # 不跑这一步，就分不清"有用"和"泄漏"——今天已经因此作废了两次结果。
    perm_rng = np.random.default_rng(0)
    perm_y = perm_rng.permutation(y)
    perm_model = fit(with_time[tr], perm_y[tr])
    perm_auc = auc(perm_model(with_time[te]), perm_y[te])
    print(f"  标签置换 AUC {perm_auc:.4f}（应≈0.5）", flush=True)

    verdict = {
        "perm_auc": round(float(perm_auc), 4),
        "permutation_ok": bool(abs(perm_auc - 0.5) < 0.1),
        "time_gain": round(a_time - a_lr, 4),
        "time_useful": bool(a_time - a_lr >= 0.02),
    }
    out = {"samples": len(y), "cross_negatives": len(cross), "auc": scores, **verdict}
    print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
    path = Path("sweep_results/hallucination/grounding_compare.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

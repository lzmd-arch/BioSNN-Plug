"""残差头的三步基线：① 合成数据 ② 规则门 ③ 逻辑回归。全部不训练 SNN。

数据集从记忆库合成：
- **正例**：把某条记忆里的一句当作"答案"，该记忆本身当证据 ⇒ 有据；
- **负例**：同一句，但把证据换成**另一条记忆** ⇒ 无据。

特征 = 五条残差里可算的四条（R2 需要工具轨迹，本数据集恒为 0，故不纳入）。

判据：报规则门与逻辑回归的 AUC 与在固定拒答率下的准确率。
**只有逻辑回归显著优于规则门，才说明"可学习的头"有价值；否则 SNN 更没有位置。**
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import urllib.parse
from pathlib import Path

import numpy as np

DB = Path("D:/AxoData/memory/mem.db")


def memories(db: Path, min_len: int = 20) -> list[str]:
    uri = "file:" + urllib.parse.quote(str(db).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "select content from memories where content is not null and length(content) > ? "
            "order by created_at",
            (min_len,),
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def r1(answer: str, evidence: str) -> float:
    items = re.findall(r"[\u4e00-\u9fff]{2,4}|\d+", answer)
    if not items:
        return 1.0
    if not evidence:
        return 0.0
    return sum(1 for i in items if i in evidence) / len(items)


def r4(answer: str, evidence: str) -> float:
    dates = re.findall(r"\d{4}年\d{1,2}月(?:\d{1,2}[日号])?|\d{1,2}月\d{1,2}[日号]", answer)
    if not dates or not evidence:
        return 0.0 if not dates else 1.0
    return sum(1 for d in dates if d not in evidence) / len(dates)


def r5(answer: str, evidence: str) -> float:
    names = re.findall(r"(?:叫|是|在|有)([\u4e00-\u9fff]{2,4})", answer)
    if not names or not evidence:
        return 0.0 if not names else 1.0
    return sum(1 for n in names if n not in evidence) / len(names)


def features(answer: str, evidence: str) -> list[float]:
    return [r1(answer, evidence), r4(answer, evidence), r5(answer, evidence), len(evidence) / 500.0]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 3000, lr: float = 0.3):
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-9] = 1.0
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    w = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z @ w)))
        w -= lr * (z.T @ (p - y) / len(y) + l2 * np.concatenate([w[:-1], [0.0]]) / len(y))
    return lambda xx: 1 / (1 + np.exp(-(np.hstack([(xx - mean) / std, np.ones((len(xx), 1))]) @ w)))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels > 0].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> int:
    rng = random.Random(0)
    texts = memories(DB)
    print(f"记忆 {len(texts)} 条", flush=True)
    rows = []
    for text in texts:
        sentences = [s for s in re.split(r"[。！？\n]", text) if 8 <= len(s) <= 60]
        if not sentences:
            continue
        claim = rng.choice(sentences)
        wrong = texts[rng.randrange(len(texts))]
        # **难的负例**：同一条记忆，只改一个数字或一个专名。
        # 整条换掉太容易（R1 一眼就看出来），真实幻觉是"证据对、细节错"。
        hard = re.sub(r"\d+", lambda m: str(int(m.group()) + rng.randint(1, 9)), text, count=1)
        if hard == text:
            names = re.findall(r"(?:叫|是|在|有)([一-鿿]{2,4})", text)
            if names:
                target = rng.choice(names)
                donor = re.findall(r"(?:叫|是|在|有)([一-鿿]{2,4})", wrong)
                if donor:
                    hard = text.replace(target, rng.choice(donor), 1)
        for evidence, label in ((text, 1), (hard if hard != text else wrong, 0)):
            rows.append({"x": features(claim, evidence), "y": label})
    print(f"样本 {len(rows)}（正 {sum(r['y'] for r in rows)}）", flush=True)

    rng.shuffle(rows)
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]
    x_tr = np.asarray([r["x"] for r in train])
    y_tr = np.asarray([r["y"] for r in train], float)
    x_te = np.asarray([r["x"] for r in test])
    y_te = np.asarray([r["y"] for r in test], float)

    # ② 规则门：只用 R1，阈值扫描取最好
    best = (-1.0, 0.0)
    for thr in np.linspace(0, 1, 21):
        pred = (x_te[:, 0] < thr).astype(float)  # R1 低 ⇒ 判"无据"
        acc = float((pred == 1 - y_te).mean())
        if acc > best[0]:
            best = (acc, float(thr))
    rule_scores = -x_te[:, 0]
    print(
        f"\n规则门（R1）：留出准确率 {best[0]:.4f}（阈值 {best[1]:.2f}），AUC {auc(rule_scores, y_te):.4f}"
    )

    # ③ 逻辑回归
    model = fit_logistic(x_tr, y_tr)
    p_te = model(x_te)
    pred = (p_te > 0.5).astype(float)
    print(f"逻辑回归：留出准确率 {float((pred == y_te).mean()):.4f}，AUC {auc(p_te, y_te):.4f}")

    out = {
        "memories": len(texts),
        "samples": len(rows),
        "rule_accuracy": round(best[0], 4),
        "rule_threshold": best[1],
        "rule_auc": round(auc(rule_scores, y_te), 4),
        "logistic_accuracy": round(float((pred == y_te).mean()), 4),
        "logistic_auc": round(auc(p_te, y_te), 4),
    }
    print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
    path = Path("sweep_results/hallucination/grounding_head.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"结果 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

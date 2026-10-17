"""前端验收口径：从图像特征里**线性读出** caption 的内容词。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/retrieval_frontend_probe_prereg.md`](../../sweep_results/retrieval_frontend_probe_prereg.md)。

## 它量什么

对每个高频内容词（如 ``dog`` / ``water`` / ``person``），训练一个**线性探针**（岭回归闭式解），
从图像特征预测"这张图的 caption 里有没有这个词"，在测试集上算 AUC。

图像特征若真的承载了 caption 描述的语义，"有没有狗"就该**线性可分**。

⚠️ **必要性而非充分性**：AUC 高只说明图像特征里有语义；**跨模态能否对齐是另一件事**，
由恢复率（`recovery.py`）衡量。两个口径各管一半。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ProbeResult",
    "auc",
    "content_words",
    "probe_words",
    "ridge_fit",
]

#: 内置停用词。**不引入外部 NLP 库**——仓库工具链与骨架库都只依赖 numpy。
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "with",
        "without",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
        "him",
        "them",
        "we",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "as",
        "into",
        "onto",
        "over",
        "under",
        "near",
        "next",
        "two",
        "three",
        "four",
        "five",
        "one",
        "some",
        "many",
        "few",
        "very",
        "more",
        "most",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "then",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "what",
        "how",
        "all",
        "any",
        "each",
    ]
)

TOKEN_RE = re.compile(r"[a-z]+")


def _tokens(caption: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(caption.lower()) if token not in STOPWORDS}


def content_words(captions: list[str], *, min_df: int = 50, top_k: int = 50) -> list[str]:
    """按文档频次取前 ``top_k`` 个内容词。"""
    counter: Counter[str] = Counter()
    for caption in captions:
        counter.update(_tokens(caption))
    keep = [(word, df) for word, df in counter.most_common() if df >= min_df]
    return [word for word, _ in keep[:top_k]]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """秩统计的 AUC。**ties 记 0.5**（与 Mann–Whitney U 一致）。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # 处理 ties：把同一分数的秩取平均
    sorted_scores = scores[order]
    start = 0
    for end in range(1, scores.size + 1):
        if end == scores.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end

    rank_sum = ranks[labels].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """闭式岭回归：``(XᵀX + λI)⁻¹ Xᵀy``，带截距（不惩罚）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.hstack([np.ones((x.shape[0], 1)), x])
    gram = design.T @ design
    penalty = lam * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0  # 截距不惩罚
    return np.linalg.solve(gram + penalty, design.T @ y)


@dataclass(frozen=True)
class ProbeResult:
    word: str
    pos_train: int
    pos_test: int
    auc: float
    lam: float


def _select_lambda(
    x_train: np.ndarray, y_train: np.ndarray, lambdas: tuple[float, ...], folds: int, seed: int
) -> float:
    if len(lambdas) == 1:
        return lambdas[0]
    rng = np.random.default_rng(seed)
    chunks = [c for c in np.array_split(rng.permutation(x_train.shape[0]), folds) if len(c)]
    best, best_score = lambdas[0], -1.0
    for lam in lambdas:
        hits = 0.0
        for fold in chunks:
            held = {int(i) for i in fold}
            keep = [i for i in range(x_train.shape[0]) if i not in held]
            weights = ridge_fit(x_train[keep], y_train[keep], lam)
            design = np.hstack([np.ones((len(held), 1)), x_train[list(held)]])
            hits += auc(design @ weights, y_train[list(held)]) * len(held)
        score = hits / x_train.shape[0]
        if score > best_score:
            best, best_score = lam, score
    return best


def probe_words(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_captions: list[str],
    test_captions: list[str],
    *,
    words: list[str] | None = None,
    min_df: int = 50,
    top_k: int = 50,
    min_pos: int = 20,
    lambdas: tuple[float, ...] = (0.1, 1.0, 10.0),
    folds: int = 5,
    seed: int = 0,
) -> tuple[list[ProbeResult], list[str]]:
    """逐词训练探针。返回结果与**被剔除的词**（正例太少）。"""
    words = (
        words if words is not None else content_words(train_captions, min_df=min_df, top_k=top_k)
    )
    train_sets = [_tokens(c) for c in train_captions]
    test_sets = [_tokens(c) for c in test_captions]

    results: list[ProbeResult] = []
    dropped: list[str] = []
    for word in words:
        y_train = np.array([word in s for s in train_sets], dtype=float)
        y_test = np.array([word in s for s in test_sets], dtype=float)
        if y_train.sum() < min_pos or y_test.sum() < min_pos:
            dropped.append(word)
            continue
        lam = _select_lambda(train_features, y_train, lambdas, folds, seed)
        weights = ridge_fit(train_features, y_train, lam)
        design = np.hstack([np.ones((test_features.shape[0], 1)), test_features])
        results.append(
            ProbeResult(
                word=word,
                pos_train=int(y_train.sum()),
                pos_test=int(y_test.sum()),
                auc=auc(design @ weights, y_test),
                lam=lam,
            )
        )
    return results, dropped

"""Cross-modal retrieval metrics: R@K for text->image and image->text.

Plan section 7's phase-two bar is "cross-modal retrieval R@1 50%+". This module is the
metric and nothing else -- it does not know what a modality plugin is, so it can be
tested without downloading anything.

## Two readings of R@1 in this repository

The bar does not say **what representation** is being retrieved over, and that choice
changes what the number means. Both are reported:

| Reading | Representation | What it measures |
| :--- | :--- | :--- |
| (a) coding quality | each plugin's own encoding | how good the encoders are |
| (b) bus acceptance | the **fused** `bus_dim` vector | whether cross-modal information survives fusion |

Section 9's "cross-modal retrieval" sits in the dual-rule-coordination phase, so **(b) is
the bar** -- it is what the spike bus and TAAF are for. But (a) is a required control:
without it, a low (b) cannot be attributed ("did fusion destroy it, or were the encoders
bad to begin with?").

## What R@K counts

For each query, rank all candidates by cosine similarity and check the rank of the
correct one. `R@1` is "the first hit is correct", `R@K` is "the correct one is in the
top K". Both retrieval directions are reported: text->image and image->text. The
headline number is the **mean of the two**, which is the convention in the literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RetrievalScores", "rank_of_correct", "recall_at_k"]


@dataclass(frozen=True)
class RetrievalScores:
    """R@K for one retrieval direction.

    Attributes:
        direction: "text2image" or "image2text".
        k: the K of R@K.
        recall: fraction of queries whose correct candidate is in the top K.
        n_queries: how many queries were scored.
    """

    direction: str
    k: int
    recall: float
    n_queries: int


def _as_matrix(x, name: str) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维 (n, d)，收到形状 {array.shape}。")
    if array.shape[0] < 1:
        raise ValueError(f"{name} 至少要有一行。")
    return array


def rank_of_correct(query: np.ndarray, gallery: np.ndarray, *, index: int) -> int:
    """Rank (1-based) of the correct gallery item for one query.

    `index` is the position of the correct item in `gallery`. Ties break toward the
    earlier gallery position -- that is what `argsort` does, and it is deterministic.
    """
    query = np.asarray(query, dtype=np.float64)
    if query.ndim != 1:
        raise ValueError(f"query 必须是一维的单条表示 (d,)，收到形状 {query.shape}。")
    gallery = _as_matrix(gallery, "gallery")
    if gallery.shape[1] != query.shape[0]:
        raise ValueError(f"查询与候选的维度必须相同，收到 {query.shape[0]} 与 {gallery.shape[1]}。")
    if not 0 <= index < gallery.shape[0]:
        raise ValueError(f"index 必须落在 [0, {gallery.shape[0]})，收到 {index}。")

    similarity = (gallery @ query) / (
        np.linalg.norm(gallery, axis=1) * np.linalg.norm(query) + 1e-12
    )
    order = np.argsort(-similarity, kind="stable")
    return int(np.where(order == index)[0][0]) + 1


def recall_at_k(
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    k: int = 1,
    direction: str = "text2image",
) -> RetrievalScores:
    """R@K over a set of paired queries and a gallery.

    The correct gallery item for query `i` is `gallery[i]` -- **the pairing is the
    identity**, i.e. row `i` of `queries` is paired with row `i` of `gallery`. That is
    the standard setup for caption-image pairs once each caption is matched to its own
    image (one caption per image here; multi-caption datasets average over captions).

    Args:
        queries: `(n, d)` query representations.
        gallery: `(m, d)` candidate representations. `m == n` is expected; if `m > n`,
            queries are paired with the first `n` gallery rows and the rest are
            distractors.
        k: the K of R@K.
        direction: label for the report.
    """
    queries = _as_matrix(queries, "queries")
    gallery = _as_matrix(gallery, "gallery")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError(
            f"查询与候选的维度必须相同，收到 {queries.shape[1]} 与 {gallery.shape[1]}。"
        )
    if gallery.shape[0] < queries.shape[0]:
        raise ValueError("候选数不能少于查询数——每个查询要有自己配对的那一个。")
    if not 1 <= k <= gallery.shape[0]:
        raise ValueError(f"k 必须落在 [1, {gallery.shape[0]}]，收到 {k}。")

    ranks = [rank_of_correct(queries[i], gallery, index=i) for i in range(queries.shape[0])]
    hits = sum(1 for rank in ranks if rank <= k)
    return RetrievalScores(
        direction=direction,
        k=k,
        recall=hits / queries.shape[0],
        n_queries=queries.shape[0],
    )

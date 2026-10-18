# 真实 50k 召回 profiling 与 MMR 优化

**日期：2026-10-18。** 数据是 `benchmarks/substrate` 的真实文本按
`scaled_memories` 放大到 50,000 条；用真正的 `MemoryStore` + `recall`，不是 SDR
层面的模拟。

## 一句话

**MMR 是召回的真热点，不是扫描。** 把逐对 `SDR.jaccard` 换成候选 0/1 矩阵后，
5,000 条单次召回 **4.19 s → 1.22 s**，金标准四项逐位不变；50,000 条从约 15 s
量级降到 **9.09 s**，剩下的瓶颈是 `_scan`（`iter_items` + 词法 Jaccard），不是 MMR。

## 5,000 条（同一库，同一查询）

| 版本 | 单次召回 |
| :--- | ---: |
| 逐对 Jaccard（旧） | 4.19 s |
| 候选矩阵（新） | **1.22 s** |

cProfile（旧版）显示 9.28 s 里 `_best_mmr` / `_max_jaccard` 占 7.7 s，其中
`np.intersect1d` 占 3.8 s、`ndarray.sort` 占 1.8 s；`_scan` 只占 1.5 s。

## 50,000 条（新 MMR 版）

| 阶段 | 耗时 |
| :--- | ---: |
| 编码 50k 词法 SDR | 80.16 s |
| 建库（50k 次 add） | 3.03 s |
| `iter_items`（解码 50k SDR） | 3.87 s |
| **一次词法召回** | **9.09 s** |
| 建倒排索引 | 13.15 s |
| 倒排 gather + Jaccard（5 查询均值） | 2.15 s（候选 50,000） |

- 50k 上倒排候选**没有缩小集合**：这份放大语料同簇拼接，几乎所有条目都共享
  公共位，候选数就是全库 50k。所以“纯词法倒排”在这份语料上不成立；真实库需要
  用稀有位的 IDF/图邻居才能缩小。
- 剩余 9 s 主要是 `_scan`：50k 次 `iter_items` 解码 + 词法 Jaccard + token 估算；
  MMR 不再是主项。
- 没有真实 5 万条记忆库；这份是同簇拼接的放大语料，数字不能直接读成真实库。

## 已落地的 Axo 改动

- `src/memcore/recall.py`：`_select` 用 `_jaccard_matrix` 预计算候选两两 Jaccard，
  `max_sim` 向量维护 MMR 的冗余度；删除了逐对的 `_best_mmr` / `_max_jaccard`。
- 语义与原实现一致：`SDR.jaccard` 刻意忽略权重，所以 0/1 矩阵同义；宽度不一致
  的候选按 0 相似度处理。
- 金标准词法四项逐位不变：hit@1 0.8961 / hit@8 0.9481 / MRR 0.9095 /
  误注入 0.8679 / AUC 0.9289。
- 已推送 Axo `main` `357e757`；相关测试与 pre-commit 全过。

## 下一步（真 50k 的剩余瓶颈）

1. **稀有位候选**：按每位 `df` 排序，先用稀有位的倒排表 gather，再补公共位；
   在 50k 放大语料上测候选缩小与 top-1 一致率。
2. **避免全量 `iter_items`**：候选确定后按 id 取条目，而不是解码全库 SDR。
3. **token 估算惰性化**：只对进入候选池的条目算，或缓存。
4. 语义/时间/图通道的候选并集：生产召回不能只按词法位过滤，否则会漏语义命中。

当前 50k 单次召回 9 s 仍然太慢；MMR 优化是必要但不充分的一步。

## 复现

```bash
uv run python scripts/axo_recall_50k_profile.py --n 5000 --queries 5
uv run python scripts/axo_recall_50k_profile.py --n 50000 --queries 5
```

## 词法倒排候选落地（Axo `bc59387`）

- `MemoryStore.lexical_index()`：按 `lex_*` 列建 CSR 倒排，按 `revision` 失效；
  `get_many()` 批量取候选条目。
- 纯词法路径（无语义 query_vector、`graph_weight<=0`、无时间窗、全部条目有词法码）
  使用候选；其他情况全量扫描。
- 50k 实测：索引首次构建 **22.3 s**；之后一次召回 **均值 1.96 s、中位 1.83 s**
  （之前 9.09 s）。5k 上候选路径 1.66 s，比矩阵版 1.22 s 略慢——候选数等于全库，
  没有收益。
- 金标准词法四项逐位不变；`tests/test_lexindex.py` 覆盖倒排、缓存失效与 `get_many`。
- 50k 的 1.96 s 仍不是终态：剩余是 5,000 条候选的 get_many + 扫描 + MMR；语义/
  时间/图通道仍走全量。下一步是并集候选，并把索引构建移出召回热路径。


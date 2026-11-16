# 本目录的结果作废（2026-11-16）

**双曲读出** —— 本目录下的 `result.json` **不能用**。

## 为什么

任务生成器 `make_task` 里的 `pref`（状态→观测的生成分布）是用
`torch.distributions.Dirichlet(...).sample()` 抽的。这个 API 走的是**全局** RNG，
不吃函数里的局部 `torch.Generator`。后果是**每次调用都抽出一个新的 `pref`**：

```python
tr = make_task(..., seed=10+sd)   # pref A
te = make_task(..., seed=20+sd)   # pref B，与 A 无关
```

于是**训练集与测试集是两个不同的任务**：`s -> token` 的映射都不一样。
任何分类器在这种测试集上都只能到随机。

## 证据：数字恰好压在随机线上

见 `result.json` 里的 `acc`（K=8 时随机线是 0.125）。它们与 0.125 的差都在
二项噪声内 —— 这不是"结构没用"，是**任务本身不成立**。

## 已修

`scripts/instrument.py` 的 `seeded_dirichlet()`（走 `torch.random.fork_rng`，
同 seed 必然同 `pref`，且不扰动调用方的全局 RNG），四个脚本已改用它。
守卫测试在 `tests/test_instrument.py`：

- `test_make_task_uses_one_pref_for_train_and_test`（行为级）
- `test_no_script_draws_dirichlet_from_the_global_rng`（静态级）

## 要做什么

**重新跑，不要引用旧数字。** 旧数字不是负结果，是没有结果。

## 重跑前要查清的一件事

本目录的 `result.json` 是 `{"cells": {"4": {}, "8": {}, "16": {}, "32": {}}}` ——
**四个 K 都是空字典**，说明主循环写完键之后在写 `out[name]` 之前就异常退出了，
而写文件发生在最后一次循环之后。也就是说这份文件不是一次完整运行的产物。

`make_task_k` 的 `pref` 未播种问题已修（`seeded_dirichlet`），但**没有证据表明
这就是空结果的原因** —— 重跑时先加 `--ks 4 --seq 16 --train-seq 8 --test-seq 4 --seeds 0`
的小配置，确认能产出完整 `cells` 再放大。

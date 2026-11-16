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

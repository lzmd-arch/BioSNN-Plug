# 本目录旧数字已作废（2026-11-16），**已重跑**

这份文件保留作历史记录。**当前 `result.json` 是重跑后的有效数据**，
判读见 [`README.md`](README.md)。

## 当时为什么作废

任务生成器 `make_task` 里的 `pref` 用 `torch.distributions.Dirichlet(...).sample()` 抽，
而这个 API 走**全局** RNG、不吃局部 `torch.Generator`。于是每次调用都抽出一个新的
`pref`：训练集与测试集是**两个不同的任务**，六臂全部压在 1/K = 0.125 上
（旧值 0.108–0.130）。

修法见 `scripts/instrument.py` 的 `seeded_dirichlet()`，守卫测试见
`tests/test_instrument.py`。

# 预注册：阶段四端到端 v5——两段式 WM → DPAR（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** v4 把 DPAR 和 WM 塞进同一个最终状态，
WM 被干扰（B R² 0.513），且预注册的 R²≥0.50 与 MSE≤0.10 本身不一致。v5 改成**两段式**：
先在 t=35 读 WM，再继续写 DPAR 到 t=85；两个探针不共享最终状态，判据也改成一致的
R²≥0.90 + MSE≤0.10。

## 任务

```text
A [0,4] · B [15,19] · Q [30,34]        → WM probe at t=35（R² 用这个状态）
DPAR K=4 对键值: key_i [40+10i,44+10i], value_i [45+10i,49+10i]
查询键 [80,84]；test_step=85            → DPAR 用每幕独立记忆；state control 用 t=85 状态
```

A/B 加 σ=0.3 噪声；OOD 为 A/B ×3（只影响 WM probe 段）。DPAR 查询键加 σ=0.1 噪声；
键/值 4 维 `N(0,I)`；K=4。

## 判据（先写死，5 种子均值）

1. WM 的 A/B/Q 平均 R² **≥0.90**；
2. WM 的 80% 覆盖选择性 MSE **≤0.10**（与 R² 一致；v3 的 0.909/0.067 是同一量级）；
3. DPAR 最好臂（Raw NN / Pseudo）top-1 **≥0.80**；
4. 状态控制 top-1 **≤0.35**（chance 0.25）；
5. OOD AUROC **≥0.75**；
6. OOD 弃权 **≥0.80**；
7. 六条全过 → `e2e_v5_ready=true`；否则报边界，不调窗/σ/K/阈值。

## 语义边界

- 这是**顺序组合**，不是 v4 的共享最终状态共存。v4 的干扰结果原样保留；v5 通过不能
  推翻 v4 的“共存会干扰 WM”。
- WM probe 在 DPAR 写入之前，因此 WM 的 R²/MSE 不受 K 对键值影响；这是设计，不是能力。
- DPAR 仍是每幕独立建库，不是终身情景记忆；OOD 仍是尺度偏移。
- 不把 `e2e_v5_ready` 写成阶段四整体验收。

## 报告义务

- WM 的 R²/MSE、DPAR 三臂、状态控制、OOD 全部单列；
- 若 WM 过但 DPAR 失败，写“顺序组合只完成 WM 段”；
- 若 OOD 弃权不过，`e2e_v5_ready=false`，不追认 v3/v4。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v5.py`；
- `sweep_results/stage4_e2e_v5/summary.json` + README。

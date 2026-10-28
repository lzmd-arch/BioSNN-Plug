# 预注册：元认知 v8——重建一致性门（2026-10-28）

**定稿时间：2026-10-28，在本轮任何 v8 最终数据之前。** v6 的量化 OOD 让全量误差升到
in-dist 的 6.38×，但 `gauss3` 门（disagreement / surprise / state_norm）弃权只有 0.172、
AUROC 0.488；v7 的高阶矩门也失效。原因是门里**没有“状态能否重建观测输入”这一维**。
v8 只改门：用读出预测与观测输入的**逐样本重建误差** + `state_norm`，做 5-NN 密度门。

## 机制

```text
state → readout → 预测 [A, B, Q, E]
errors = mean((预测 − 观测输入)²)          ← 推理时输入可观测，不是标签泄漏
state_norm = ‖state‖
门：在 calibration 上标准化 [errors, state_norm]，5-NN 距离取负为置信度
阈值：calibration 置信度的 20% 分位（→ 80% 覆盖）
```

- 量化 OOD：输入幅度丢失，state 仍“像” in-dist，但重建误差升高；误差维负责检出。
- 尺度 OOD：state_norm 与重建误差同时升高；`state_norm` 维负责 scale sanity。
- 两个维度的组合是**重建一致性**，不是把测试误差当标签：观测输入在推理时本来就有。

## 判据（与 v6 相同，先写死）

1. **有效性**：quantized 全量 MSE ≥2× in-dist 全量 MSE；
2. 测试覆盖 0.78–0.82（阈值只在 calibration 上取）；
3. 选择性 MSE ≤0.10；
4. quantized OOD 弃权 ≥0.80；
5. quantized OOD AUROC ≥0.75；
6. scale sanity 弃权 ≥0.80；
7. 六条全过 → `v8_ready`；否则报边界，不调 k/特征/阈值/判据。

## 数据独立性（先声明）

- 预注册前的门族选择用**种子 0–4**（试点：kNN[errors,state_norm] 量化弃权 0.912、
  AUROC 0.927、覆盖 0.799、scale 0.822）；**最终结果用种子 5–9**，两者不混报。
- 预注册之后不改 k=5、不改特征、不改判据。

## 边界（预先写死）

- v8 只覆盖“输入被变换后 state 与输入不一致”的 OOD（量化、尺度）；不覆盖标签漂移、
  状态零空间内的变换、或与输入一致的分布偏移；不能写成通用 OOD。
- `errors` 是重建一致性，不是独立自编码器；它是同一个线性读出的预测误差。推理时观测
  输入可得，因此不是标签泄漏；但若部署时输入不可得，这个门不适用。

## 不做什么

- 不调邻居数、特征、阈值分位、覆盖率、判据；
- 不把 v6/v7 的负结果混进 v8；scale 与 quantized 都报；
- 不把 v8 写成“通用弃权”。

## 产物

- `research/cognitive/run_metacognitive_gate_v8.py`；
- `sweep_results/metacognitive_gate_v8/summary.json` + README。

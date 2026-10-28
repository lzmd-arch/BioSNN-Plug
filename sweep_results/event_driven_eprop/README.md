# 事件驱动 e-prop：最小实现（LIF / 最后一步损失 / 固定 fan-in）

**性质：实现，不是规模实验。** 目标是把 O(N²) 的资格痕迹换成 O(N·k)，先把
10k–100k 的可行性做出来；本轮没有跑大规模训练。

## 实现

- `research/eprop/event_driven.py`：`EventDrivenEProp`
- 限定：**LIF（β=0）**、**损失只取最后一步**、**固定稀疏 fan-in k**（不含自连接）；
- 在这些条件下，ε_v 与后突触下标无关，梯度可写成
  `outer(filtered_pre, L_j·ψ_j)`，所以：
  - 权重 O(N·k)；
  - 前突触滤波脉冲 O(N)；
  - 不存在 O(N²) 的 ε_a/梯度矩阵。

## 正确性

`research/eprop/tests/test_event_driven.py`：在**同一稀疏 mask** 下，与现有 dense
`EPropLearner`（β=0、最后一步、相同 w_in/w_out/w_rec）对照，w_in / w_out / 被 mask
覆盖的 w_rec 更新逐位一致（atol 1e-6）。

## 边界（不夸大）

- 前向仍按时间步推进；这不是 Korcsák-Gorzo 的完整事件驱动实现，省掉的是 O(N²)
  痕迹，不是前向 kernel；
- 只支持最后一步损失；`loss_timesteps="all"` 不在本实现内；
- 稀疏 mask 固定，不学习连接；
- 没有跑 10k/100k 训练，所以不能写“已扩展到 500k”。

## 下一步

1. 在 sMNIST 上跑 n_rec=4k/8k 的 1–2 epoch 规模冒烟，量显存/耗时；
2. 若可行，再把固定 fan-in 与事件驱动前向接上；
3. 与 dense 的精度对照必须在同预算、同种子下做，不混报。

## 产物

- `research/eprop/event_driven.py`；
- `research/eprop/tests/test_event_driven.py`。

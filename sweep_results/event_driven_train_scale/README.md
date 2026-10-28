# 事件驱动 e-prop：单 batch 训练冒烟（2026-10-29）

**性质：一次 `update` 的路径/时间冒烟，不是 epoch 训练，也不报精度。** 输入 n_in=28、
T=28、batch=1、fan_in=32、n_out=10；脚本 `scripts/measure_event_driven_train.py`。

## 结果

| n_rec | 稀疏权重 | 一次 update |
| ---: | ---: | ---: |
| 10,000 | 320,000 | **0.10 s** |
| 100,000 | 3,200,000 | **0.88 s** |
| 500,000 | 16,000,000 | **5.27 s** |

loss 分别为 2.1902 / 2.2890 / 2.6353（随机初始化 + 单 batch，**不作准确率/收敛结论**）。

## 能说什么 / 不能说什么

- **能说**：固定 fan-in 的事件驱动 **update 路径**在 500k、batch=1 下可跑通；
- **不能说**：500k 的 epoch 训练可跑——单 batch 5.27 s，按 60k 样本/batch=1 外推约 90 h，
  且没有测 batch>1 的峰值；
- 不能把 loss 2.x 当模型质量；没有训练曲线、没有测试集；
- 阶段四规模目标仍未验收；下一步是 batch>1 的峰值/吞吐，以及 10k 的真实 1 epoch。

## 下一步

1. 10k、batch=8 的 1 epoch（sMNIST）量时间/内存；
2. 100k 只在 batch 小、样本子集上做可行性；
3. 500k 训练暂不承诺，前向与单 update 已证明路径可行。

## 自证

`summary.json` 由脚本生成；命令：
`python scripts/measure_event_driven_train.py --sizes 10000,100000,500000`。

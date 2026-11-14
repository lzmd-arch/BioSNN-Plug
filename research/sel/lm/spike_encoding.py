"""**时间编码**：把连续量变成"谁先发放"——这是 SNN 自己的编码，不是查表加投影。

## 为什么

现状：`emb(token) → proj → 加性电流`，一步到位，与 transformer 同构；τ 在数 token。

本模块用**首脉冲延迟（time-to-first-spike）**编码一个连续向量 c：
通道 i 的值越大，它发放得越早；值越小，越晚（或整段不发放）。

```
latency_i = round(T · (1 − c_i))     # c 归一化到 [0,1]，T 是编码窗
```

每一步只把"该在这一步发放"的那些通道置原位，其余为 0。于是**信息落在时间轴上**，
而 SNN 的膜电位衰减正好对这些时刻敏感。

## 边界

- 只用于**新的连续模态**（传感器/行为），不动文本通路；
- 每个通道单独发放一次，不重复（避免变成速率编码）。
"""

from __future__ import annotations

import numpy as np


def normalize(vector, lo: float = 0.0, hi: float = 1.0):
    """归一到 [0,1]。**默认用固定量程 [0,1]，不做逐样本 min-max。**

    逐样本归一化会把最小值恒映射到 0 ⇒ 那个通道永远不发放，
    于是"少了一个通道"这件事与数值无关、只与它是不是最小有关——信息被伪造了。
    量程应当由数据本身定（传感器有各自的物理范围），不是由这一批样本定。
    """
    x = np.asarray(vector, dtype=np.float32)
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def latencies(vector, steps: int = 8, threshold: float = 0.05):
    """每个通道的发放步号。值 ≤ threshold 的通道不发放（返回 None）。"""
    c = normalize(vector)
    out = []
    for value in c:
        if value <= threshold:
            out.append(None)
        else:
            out.append(round((1.0 - value) * (steps - 1)))
    return out


def encode(vector, steps: int = 8, threshold: float = 0.05):
    """返回长度为 steps 的帧序列，每帧是与 vector 同维的稀疏向量。"""
    c = normalize(vector)
    lat = latencies(vector, steps, threshold)
    frames = np.zeros((steps, c.size), dtype=np.float32)
    for channel, step in enumerate(lat):
        if step is not None:
            frames[step, channel] = c[channel]
    return frames

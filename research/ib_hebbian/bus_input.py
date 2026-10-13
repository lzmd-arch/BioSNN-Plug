"""把脉冲总线的输出喂进 W1：`(T, N)` 脉冲 → `(m, n_in)` 连续特征。

**这是「感知层接入脉冲总线」的映射层**（计划书 §七 第二阶段任务）。W1 是
**速率型** LReLU 网络（Pogodin & Latham 的网络本身就是速率型，换成脉冲会脱离来源
论文——计划书原文），所以「脉冲进、层内速率运算」要求在入口处把脉冲**聚合成速率**。
本模块做的就是这一件事，不多做。

⚠️ **这个聚合是本项目自己的选择，不是论文的规定。** 与
:mod:`research.eprop` 的已知边界第 5 条（「输入编码是本项目自己的选择」）同一条性质。
它的取舍：

* 优点：脉冲表示真的进了 W1，`biosnn-bus` 的插件被真实消费，而不是被绕开；
* 代价：发放率是 ``T`` 个时间步上的均值，故还原误差 ``O(1/sqrt(T))``——
  ``T`` 越大越准，**不是**无损还原。
  要无损路径请用 :func:`to_features_from_raw`（直接读 ``.npy``，走 W1 原来的
  ``(x/255 − 0.5) / 0.5`` 预处理）。

两条路径的对照不是小事：W1 的验收数字 **0.9779** 是在**连续输入**上拿到的，改走脉冲
路径**未必**还等于它。所以 `train_mnist.py` 的默认仍是连续输入，脉冲路径要显式开启
（``--input-path bus``），两条路径都留得住、可对照。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from biosnn_bus import SpikeTrain

__all__ = ["encode_batch", "spike_train_to_features", "to_features_from_raw"]

#: 论文 D.4 的 MNIST 预处理常量，与 `research/ib_hebbian/model.py` 保持一致。
MNIST_SHIFT = 0.5
MNIST_SCALE = 0.5


def spike_train_to_features(
    train: SpikeTrain,
    *,
    batch_size: int,
    n_in: int,
    n_pixels: int | None = None,
) -> torch.Tensor:
    """把一条 `SpikeTrain` 的 `T` 个时间步翻成 `batch_size` 条 `n_in` 维特征。

    **时间轴 → 样本轴**：`train` 是单条序列（`(T, N)`），这里按 `batch_size` 把它
    切成 `batch_size` 段、每段一条特征。若 `T` 不能被 `batch_size` 整除，最后一段
    补零对齐——**丢掉的是最后一段不足的时间步**，不是样本身份。

    ⚠️ 但更常见的用法是 `encode_batch`：它给每张图各自编码一条序列，而不是把一整批
    图拼成一条长序列。本函数存在的意义是**同构**：总线的 `step()` 就是返回一条
    `(T, N)`，本函数把它还原成一批特征，调用方不必知道"那条序列里含几张图"。

    Args:
        train: 形状 `(T, N)` 的脉冲/电流序列。
        batch_size: 要还原成多少条特征。
        n_in: 特征维度（W1 是 28*28 = 784）。
        n_pixels: 有效像素数；`None` 时取 `N`。当 `N > n_in`（例如 ON/OFF 双通道
            的 `2*H*W`）时，取前 `n_in` 维——多出来的部分在这里不参与。

    Returns:
        形状 `(batch_size, n_in)` 的 `float32` 张量，取值在 `[0, 1]`。
    """
    if not isinstance(train, SpikeTrain):
        raise TypeError(f"需要 SpikeTrain，收到 {type(train).__name__}。")
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须是正整数，收到 {batch_size!r}。")

    data = np.asarray(train.data, dtype=np.float64)
    t_steps, n_neurons = data.shape
    width = n_pixels if n_pixels is not None else n_neurons
    if width < n_in:
        raise ValueError(
            f"序列只有 {n_neurons} 个神经元，取不出 {n_in} 维特征；"
            "这多半是 SpikeBus 的 bus_dim 与 W1 的 n_in 对不上。"
        )

    # 按时间轴切段：每段的发放率就是该段的特征。
    edges = np.linspace(0, t_steps, batch_size + 1, dtype=int)
    rows = []
    for i in range(batch_size):
        lo, hi = int(edges[i]), int(edges[i + 1])
        # 时间步不够切：补零，不伪造特征。
        rate = data[lo:hi, :n_in].mean(axis=0) if hi > lo else np.zeros(n_in, dtype=np.float64)
        rows.append(rate)
    return torch.from_numpy(np.stack(rows, axis=0)).to(torch.float32)


def encode_batch(
    plugin: Any,
    raw: Any,
    *,
    batch_size: int,
    n_in: int,
) -> torch.Tensor:
    """把一批原始模态输入经**插件自己**编码再解码 → `(batch_size, n_in)` 特征。

    ⚠️ **这里刻意不经 `SpikeBus.step`。** `step()` 的输出是融合后送进认知核心的
    **输入电流**（稀疏随机投影带 ±1 权重），**不是 0/1 脉冲**；拿它的输出当脉冲去
    平均，等于把投影矩阵的 ±1 权重算进了特征——这是把两种不同的量混成一个。
    本模块要的是"脉冲表示"，所以直接用插件的 `encode` / `decode` 这一对互逆操作，
    让投影只在多模态融合时才出现。

    逐条编码：每张图各自产出一条 `(T, N)`，再用插件自己的 `decode` 聚合成一条特征。
    这样样本之间互不污染（不是把一整批拼成一条长序列再切）。
    """
    frames = np.asarray(raw, dtype=np.float64)
    if frames.ndim != 3:
        raise ValueError(f"encode_batch 期望 (m, H, W) 的批量输入，收到 {frames.shape}。")
    if frames.shape[0] != batch_size:
        raise ValueError(f"batch_size={batch_size} 与输入的第一维 {frames.shape[0]} 不一致。")

    rows = []
    for i in range(batch_size):
        train = plugin.encode(frames[i])
        flat = np.asarray(plugin.decode(train), dtype=np.float64).reshape(-1)
        if flat.size < n_in:
            raise ValueError(f"插件解码只得到 {flat.size} 维，少于 n_in={n_in}。")
        rows.append(flat[:n_in])
    return torch.from_numpy(np.stack(rows, axis=0)).to(torch.float32)


def to_features_from_raw(images: torch.Tensor) -> torch.Tensor:
    """W1 原来的连续输入路径：`(m, 28, 28)` / `(m, 784)` → `(m, 784)` 取值 `[-1, 1]`。

    逐字对应 `research/ib_hebbian/model.py` 的 `preprocess_mnist`（论文 D.4）。
    本函数只是给 `--input-path raw` 一个同名入口，**不改任何数值行为**。
    """
    x = images.reshape(images.shape[0], -1).to(torch.float32) / 255.0
    return (x - MNIST_SHIFT) / MNIST_SCALE

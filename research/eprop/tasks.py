"""顺序任务：sMNIST（主验收）与 SHD（旁证）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么是 sMNIST 做主验收

计划书 §七 只写"顺序任务可用"，没指定数据集。选 **sequential-MNIST** 的理由：

* **与 W1 复用同一份数据**——`scripts/download_data.py mnist` 下过一次就够，不必再引入
  第二个下载源；
* **判别信息真正在时间上展开**：28 行像素按行喂入，最后一行才给出完整图像，所以这个
  任务确实考验时序记忆，不是把一张图一次性喂进去；
* **CPU 上也跑得动**，CI 的冒烟测试能覆盖。

**SHD 是旁证**：Bellec et al. 2020 与 Pes et al. 2025 都用它，数字可与引文对标。但它要
另外下载（约 130 MB），且是 HDF5 格式，所以放在 :func:`load_shd` 里单独走。

## 模拟值怎么变成脉冲

sMNIST 的像素是 0–255 的模拟值，而这里的输入必须是脉冲。做法是把归一化后的像素当作
**发放概率**做伯努利采样：``x ~ Bernoulli(pixel/255)``。

这是一个**编码选择**，不是论文的规定——换个编码（例如把像素值当恒定电流、或做成
首次脉冲时刻编码）结果会不同。所以种子必须固定并可复现，且这里的选择要写进结论的边界。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["SMNIST", "load_shd", "load_smnist"]

#: sMNIST 的时间步数 = 图像行数。
SMNIST_STEPS = 28


class SMNIST:
    """sequential-MNIST 序列。

    Attributes:
        inputs: ``(T=28, n_samples, 28)`` 的脉冲张量。
        labels: ``(n_samples,)`` 的类别下标。
    """

    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def batch(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[:, index], self.labels[index]


def load_smnist(data_dir: str | Path, *, split: str, seed: int) -> SMNIST:
    """把 MNIST 展成逐行喂入的脉冲序列。

    Args:
        data_dir: ``scripts/download_data.py`` 的落盘目录。
        split: ``"train"`` 或 ``"test"``。
        seed: 伯努利采样的种子。**编码是随机的**，所以种子是复现的一部分。

    Returns:
        :class:`SMNIST`。
    """
    if split not in ("train", "test"):
        raise ValueError(f"split 只能是 'train' 或 'test'，收到 {split!r}。")

    root = Path(data_dir) / "mnist"
    images_path = root / f"{'train' if split == 'train' else 'test'}_images.npy"
    labels_path = root / f"{'train' if split == 'train' else 'test'}_labels.npy"
    if not images_path.exists() or not labels_path.exists():
        raise SystemExit(
            f"缺少 {images_path} 或 {labels_path}。\n"
            f"请先运行：uv run python scripts/download_data.py mnist"
        )

    images = np.load(images_path).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    labels = np.load(labels_path).astype(np.int64)

    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.from_numpy(images).reshape(-1, SMNIST_STEPS, 28)
    # (n, T, 28) → (T, n, 28)：第 t 步给出第 t 行像素
    spikes = (torch.rand(probabilities.shape, generator=generator) < probabilities).float()
    return SMNIST(spikes.transpose(0, 1).contiguous(), torch.from_numpy(labels))


def load_shd(data_dir: str | Path, *, split: str, seed: int, max_time: int = 100) -> SMNIST:
    """加载 SHD（Spiking Heidelberg Digits）——**旁证**数据集。

    需要先有 ``data/shd/{train,test}.h5``。当前 ``scripts/download_data.py`` 尚未登记它
    （它随本文件一起落地，见那里的注释），所以这里先给出明确的失败信息，而不是留半截实现。

    SHD 的原始格式是每段音频的脉冲时刻列表，需要栅格化成固定长度的时间窗。这个转换是
    本项目自己的选择（``max_time`` 与时间分箱），不是数据集的规定——结论里要标明。
    """
    import h5py

    path = Path(data_dir) / "shd" / f"{split}.h5"
    if not path.exists():
        raise SystemExit(
            f"缺少 {path}。SHD 需要单独下载（约 130 MB）：\n"
            f"  https://zenkelab.org/datasets/  → shd_train.h5.gz / shd_test.h5.gz\n"
            f"  解压后放到 {path.parent}/ 下，命名为 train.h5 / test.h5。\n"
            f"本仓库不提供数据镜像（计划书 §12.1）。"
        )

    with h5py.File(path, "r") as handle:
        times = handle["spikes"]["times"]
        units = handle["spikes"]["units"]
        labels = handle["labels"][:]

    n_samples = len(labels)
    n_units = 700
    generator = torch.Generator().manual_seed(seed)
    raster = torch.zeros(n_samples, max_time, n_units)
    for index in range(n_samples):
        bins = np.minimum((times[index] * 1000).astype(np.int64), max_time - 1)
        np.add.at(raster[index].numpy(), (bins, units[index].astype(np.int64)), 1.0)
    _ = generator
    raster = (raster > 0).float()

    return SMNIST(raster.transpose(0, 1).contiguous(), torch.from_numpy(labels.astype(np.int64)))

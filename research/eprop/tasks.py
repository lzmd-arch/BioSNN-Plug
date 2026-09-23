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
另外下载（两个文件合计约 169 MB），且是 HDF5 格式，所以放在 :func:`load_shd` 里单独走。

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


#: SHD 的固定时间窗（秒）。发布集里最长的一段是 1.369 s（train），取 1.4 s 把全部脉冲
#: 都收进来、不足的留空。**不做逐样本归一化**——那会把快慢不同的发音拉成同一节奏。
SHD_WINDOW_S = 1.4


def load_shd(data_dir: str | Path, *, split: str, seed: int, max_time: int = 100) -> SMNIST:
    """加载 SHD（Spiking Heidelberg Digits）——**旁证**数据集。

    需要先有 ``data/shd/{train,test}.h5``——由 ``scripts/download_data.py shd`` 落盘
    （下载约 165 MB 的 ``shd_{train,test}.h5.gz`` 并解压；本仓库不提供数据镜像，计划书 §12.1）。

    SHD 的原始格式是每段音频的脉冲时刻列表，需要栅格化成固定长度的时间窗。这个转换是
    本项目自己的选择（``max_time`` 与时间分箱），不是数据集的规定——结论里要标明。
    """
    import h5py

    path = Path(data_dir) / "shd" / f"{split}.h5"
    if not path.exists():
        raise SystemExit(
            f"缺少 {path}。先跑：\n"
            f"  uv run python scripts/download_data.py shd\n"
            f"它会下载 shd_train.h5.gz / shd_test.h5.gz（约 165 MB）并解压到 {path.parent}/。\n"
            f"本仓库不提供数据镜像（计划书 §12.1）。"
        )

    # **可变长数据必须在 with 块内取出来。** h5py 的 vlen 数据集是按需读取的，出了
    # with（文件已关）再索引就是 `RuntimeError: Unable to synchronously get dataspace`。
    # 这个缺陷在本函数第一次被真正运行之前一直藏着——它此前从未跑过，见 README 的边界。
    with h5py.File(path, "r") as handle:
        labels = handle["labels"][:]
        times = list(handle["spikes"]["times"])
        units = list(handle["spikes"]["units"])

    n_samples = len(labels)
    n_units = 700
    raster = np.zeros((n_samples, max_time, n_units), dtype=np.float32)
    for index in range(n_samples):
        # **把整段音频铺到 max_time 个箱里**（每箱 14 ms），不是只取开头 100 ms。
        # 发布集的脉冲时刻最远到 1.37 s，按毫秒直接截断会丢掉绝大部分脉冲。
        bins = np.minimum((times[index] / SHD_WINDOW_S * max_time).astype(np.int64), max_time - 1)
        np.add.at(raster[index], (bins, units[index].astype(np.int64)), 1.0)

    # 二值化：这个任务里"某箱内有脉冲"比"有几颗"更接近论文的输入约定；`seed` 不参与——
    # 与 sMNIST 不同，这里的脉冲时刻是数据给定的，没有随机编码步骤。参数保留是为了与
    # :func:`load_smnist` 的签名对称。
    _ = seed
    spikes = torch.from_numpy((raster > 0).astype(np.float32))
    return SMNIST(spikes.transpose(0, 1).contiguous(), torch.from_numpy(labels.astype(np.int64)))

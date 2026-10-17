"""程序生成的多模态配对集：图像 ↔ 描述。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

这个数据集存在的理由只有一个：**在没有任何下载与许可问题的前提下，先回答
"这条管线能不能把两路模态对齐"**。它不是基准，不能用来对标文献——
真正的对照（Flickr8k）有加载器但没有在本轮运行，理由见
``sweep_results/retrieval_procedural_prereg.md``。

## 设计（与预注册一致）

四个属性组合出 144 个样本，每个样本一条 caption，配对是**同一性**：

| 属性 | 取值 |
| :--- | :--- |
| 形状 | square / circle / triangle / cross |
| 明度 | dark / medium / bright（灰度 0.35 / 0.65 / 1.0） |
| 位置 | 3×2 网格，六个方位 |
| 尺寸 | small / large（边长 4 / 6，画布 24×24） |

**图像是帧序列，不是单帧**：`image_diff` 编码的是**相邻帧的差**，所以静止的单帧
会得到全零脉冲。这里用「空帧 → 图像 → 图像」三帧，让第一帧跳变产生 ON 事件。

⚠️ **属性是独立均匀枚举的**，所以 caption 里每个 token 的分布可控、组合空间是
**可数的**——这正是能算出精确随机基线的原因（1/测试集条数），也是它不能代替
真实数据的代价。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

__all__ = ["CANVAS", "FRAMES", "Item", "generate", "split"]

#: 画布边长（像素）。
CANVAS = 24
#: 每张图编成的帧数：空帧 + 两帧图像，让 DVS 编码器看到一次跳变。
FRAMES = 3

SHAPES = ("square", "circle", "triangle", "cross")
SHADES = (("dark", 0.35), ("medium", 0.65), ("bright", 1.0))
POSITIONS = (
    ("top left", (0, 0)),
    ("top centre", (0, 1)),
    ("top right", (0, 2)),
    ("bottom left", (1, 0)),
    ("bottom centre", (1, 1)),
    ("bottom right", (1, 2)),
)
SIZES = (("small", 4), ("large", 6))


@dataclass(frozen=True)
class Item:
    """一条配对样本。"""

    image: np.ndarray
    """``(FRAMES, CANVAS, CANVAS)`` 的灰度帧序列，取值 [0, 1]。"""
    caption: str
    shape: str
    shade: str
    position: str
    size: str

    @property
    def attrs(self) -> tuple[str, str, str, str]:
        return (self.shape, self.shade, self.position, self.size)


def _draw(shape: str, side: int, value: float, row: int, col: int) -> np.ndarray:
    """在 3×2 网格的第 ``(row, col)`` 格里画一个 ``shape``，返回单帧。"""
    frame = np.zeros((CANVAS, CANVAS), dtype=np.float64)
    cell_h = CANVAS // 2
    cell_w = CANVAS // 3
    top = row * cell_h + (cell_h - side) // 2
    left = col * cell_w + (cell_w - side) // 2

    rr, cc = np.mgrid[0:side, 0:side]
    if shape == "square":
        mask = np.ones((side, side), dtype=bool)
    elif shape == "circle":
        centre = (side - 1) / 2.0
        mask = (rr - centre) ** 2 + (cc - centre) ** 2 <= (side / 2.0) ** 2
    elif shape == "triangle":
        mask = rr >= (side - 1) - cc
    elif shape == "cross":
        arm = max(1, side // 3)
        mid = side // 2
        mask = (np.abs(rr - mid) <= arm) | (np.abs(cc - mid) <= arm)
    else:  # pragma: no cover - 由构造保证不可达
        raise ValueError(f"未知形状：{shape}")

    frame[top : top + side, left : left + side][mask] = value
    return frame


def generate() -> list[Item]:
    """按固定顺序枚举全部 144 个组合。顺序固定是为了可复现。"""
    items: list[Item] = []
    for shape, (shade, value), (position, (row, col)), (size, side) in product(
        SHAPES, SHADES, POSITIONS, SIZES
    ):
        frame = _draw(shape, side, value, row, col)
        # 空帧 → 图像 → 图像：第一步产生 ON 跳变，之后静止。
        image = np.stack([np.zeros_like(frame), frame, frame], axis=0)
        caption = f"a {size} {shade} {shape} in the {position} of the image"
        items.append(
            Item(
                image=image,
                caption=caption,
                shape=shape,
                shade=shade,
                position=position,
                size=size,
            )
        )
    return items


def split(
    items: list[Item], *, test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Item], list[Item]]:
    """按种子随机切出测试集。

    **切分只做一次、在拟合之前**——训练集不参与任何验收指标，测试集不参与任何选择。
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction 必须落在 (0, 1)，收到 {test_fraction}")
    order = np.random.default_rng(seed).permutation(len(items))
    n_test = max(1, round(len(items) * test_fraction))
    test_idx = {int(i) for i in order[:n_test]}
    train = [item for i, item in enumerate(items) if i not in test_idx]
    test = [item for i, item in enumerate(items) if i in test_idx]
    return train, test

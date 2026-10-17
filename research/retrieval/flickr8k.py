"""Flickr8k 加载器：真实图文配对，供跨模态检索的**第二个**数据集使用。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 它现在做到哪一步

**只做加载，不做下载。** 下载由 `scripts/download_data.py flickr8k` 负责（计划书 §12.1
的「不镜像数据，给下载脚本 + 逐文件校验」）：源是 **jbrownlee/Datasets 的 GitHub Release**
（非官方镜像，稳定直链），两个 zip 的 MD5 已钉在该脚本里。下完解压到这个结构：

```text
data/flickr8k/
├─ Flickr8k.token.txt                     # image.jpg#0<TAB>caption
├─ Flickr_8k.trainImages.txt              # 每行一个 image.jpg
├─ Flickr_8k.devImages.txt
├─ Flickr_8k.testImages.txt
└─ Flicker8k_Dataset/                     # 注意官方目录名就是 Flicker（少一个 r）
   ├─ 1000268201_693b08cb0e.jpg
   └─ ...
```

下面是官方结构本身（供人工摊放时对照）：

```text
<root>/
├─ Flickr8k.token.txt                     # image.jpg#0<TAB>caption
├─ Flickr_8k.trainImages.txt              # 每行一个 image.jpg
├─ Flickr_8k.devImages.txt
├─ Flickr_8k.testImages.txt
└─ Flicker8k_Dataset/                     # 注意官方目录名就是 Flicker（少一个 r）
   ├─ 1000268201_693b08cb0e.jpg
   └─ ...
```

⚠️ 目录名 `Flicker8k_Dataset` 少一个 `r`——这是官方压缩包里的写法，不是笔误。
改成 `Flickr` 会让加载器找不到图，所以这里按原样认。

## 图像怎么进管线

和程序生成集同一个约定：**空帧 → 图像 → 图像**三帧，让 `image_diff`（DVS，编码帧差）
看到一次跳变。灰度、缩放到 24×24、归一到 [0, 1]。

⚠️ **每张图五条 caption，这里只取一条**（默认第 0 条，固定顺序）。取哪一条写在记录里；
取多条要改检索口径（一对多），那是另一个协议，不在本轮范围。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .procedural import CANVAS

__all__ = ["DATASET_DIR", "FlickrItem", "load_captions", "load_split", "make_items"]

#: 官方压缩包里的目录名，**少一个 r**。
DATASET_DIR = "Flicker8k_Dataset"


@dataclass(frozen=True)
class FlickrItem:
    """一条图文配对。字段与 :class:`research.retrieval.procedural.Item` 的
    ``image`` / ``caption`` 同名，所以两种数据集能直接喂给同一个编码器。"""

    image: np.ndarray
    caption: str
    image_id: str
    path: Path | None = None
    """原图路径。冻结预训练前端（ResNet）要读原图，而不是 24×24 的帧序列。"""


def load_captions(root: Path | str) -> dict[str, list[str]]:
    """解析 ``Flickr8k.token.txt``，返回 ``image.jpg -> [caption, ...]``。

    格式：``image.jpg#0\tA child in a pink dress...``，每张图五行，编号 0–4。
    """
    path = Path(root) / "Flickr8k.token.txt"
    if not path.is_file():
        raise FileNotFoundError(f"缺 {path}——先按模块文档的结构把 Flickr8k 摊好。")
    captions: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, _, text = line.partition("\t")
        image_id = name.split("#", 1)[0]
        captions.setdefault(image_id, []).append(text.strip())
    if not captions:
        raise ValueError(f"{path} 解析出 0 条 caption——格式不对？")
    return captions


def load_split(root: Path | str, split: str) -> list[str]:
    """读 ``Flickr_8k.<split>Images.txt``，返回 image 文件名列表。"""
    mapping = {
        "train": "Flickr_8k.trainImages.txt",
        "dev": "Flickr_8k.devImages.txt",
        "test": "Flickr_8k.testImages.txt",
    }
    if split not in mapping:
        raise ValueError(f"split 必须是 {sorted(mapping)} 之一，收到 {split!r}")
    path = Path(root) / mapping[split]
    if not path.is_file():
        raise FileNotFoundError(f"缺 {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _frames(path: Path) -> np.ndarray:
    """JPEG → ``(FRAMES, CANVAS, CANVAS)`` 灰度帧序列（空帧 + 两帧图像）。"""
    from PIL import Image

    with Image.open(path) as handle:
        gray = handle.convert("L").resize((CANVAS, CANVAS))
        frame = np.asarray(gray, dtype=np.float64) / 255.0
    return np.stack([np.zeros_like(frame), frame, frame], axis=0)


def make_items(
    root: Path | str,
    *,
    split: str = "test",
    caption_index: int = 0,
    limit: int | None = None,
    decode_frames: bool = True,
) -> list[FlickrItem]:
    """把某个 split 摊成配对列表。取不到的图**跳过并计数**，不静默吞掉。

    ``decode_frames=False`` 只记录路径、不把 JPEG 解成 24×24 帧序列——冻结前端要的是
    原图，解一遍帧是白费（7000 张要几十秒到几分钟）。
    """
    root = Path(root)
    captions = load_captions(root)
    ids = load_split(root, split)
    if limit is not None:
        ids = ids[:limit]

    items: list[FlickrItem] = []
    missing = 0
    for image_id in ids:
        path = root / DATASET_DIR / image_id
        if not path.is_file() or image_id not in captions:
            missing += 1
            continue
        captions_for = captions[image_id]
        if caption_index >= len(captions_for):
            raise IndexError(
                f"{image_id} 只有 {len(captions_for)} 条 caption，取不到第 {caption_index} 条。"
            )
        items.append(
            FlickrItem(
                image=_frames(path) if decode_frames else np.zeros((0, CANVAS, CANVAS)),
                caption=captions_for[caption_index],
                image_id=image_id,
                path=path,
            )
        )
    if missing:
        print(f"⚠️ {missing} / {len(ids)} 条样本缺图或缺 caption，已跳过。")
    return items

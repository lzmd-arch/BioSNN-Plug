"""冻结预训练前端：ResNet18（图像）与 nomic（整句）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 边界

仓库允许**冻结的预训练前端**（README「感知层允许冻结预训练前端」）：判据是**参数进不进
梯度**，不是模型大不大。这里两个前端都在 ``torch.no_grad()`` 下前向，权重不参与任何训练。

⚠️ **两端不共享预训练对齐**：ResNet18 与 nomic 从未一起训练，所以"把两个模态接起来"
这件事仍然完全落在联想层。换成 CLIP 同时编码图与文会让对齐由预训练完成、局部规则无事可做，
那才是作弊——这里刻意不这么做。

## 为什么必须缓存

ResNet 要读 7000 张原图、Ollama 要跑 7000 条整句——都慢。特征落到
``data/flickr8k/`` 下，第二次运行直接读盘。缓存文件里写了前端标识（模型名与预处理），
不匹配就报错，避免"换了前端却读到旧特征"这种静默错误。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["RESNET_DIM", "TEXT_DIM", "resnet_features", "text_features"]

RESNET_DIM = 512
TEXT_DIM = 768


def _l2(array: np.ndarray) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-12)


def resnet_features(
    paths: list[Path], *, cache: Path | None = None, batch_size: int = 64
) -> np.ndarray:
    """ResNet18（ImageNet 权重，冻结）→ ``(n, 512)``，**已 L2 归一化**。"""
    if cache is not None and cache.is_file():
        payload = np.load(cache, allow_pickle=False)
        if payload["tag"].item() == "resnet18-imagenet1k-v1/224/avgpool" and payload[
            "n"
        ].item() == len(paths):
            print(f"  特征缓存命中 {cache.name}")
            return payload["features"]

    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )

    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = paths[start : start + batch_size]
            tensors = []
            for path in batch:
                with Image.open(path) as handle:
                    tensors.append(preprocess(handle.convert("RGB")))
            features = model(torch.stack(tensors).to(device))
            rows.append(features.detach().cpu().numpy().astype(np.float64))
            done = min(start + batch_size, len(paths))
            if done % (batch_size * 10) == 0 or done == len(paths):
                print(f"  ResNet {done}/{len(paths)}", end="\r")
    print()
    features = _l2(np.concatenate(rows, axis=0))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            features=features,
            n=np.array(len(paths)),
            tag=np.array("resnet18-imagenet1k-v1/224/avgpool"),
        )
    return features


def text_features(
    captions: list[str], *, model: str = "nomic-embed-text", cache: Path | None = None
) -> np.ndarray:
    """整句 nomic embedding → ``(n, 768)``，**已 L2 归一化**。

    ⚠️ 这里是**整句**编码，不是 `text_zscore` 插件那种逐 token。nomic 是句子模型，
    逐 token 编码会丢掉上下文——那条代价写在 `text_zscore.py` 的模块文档里。
    """
    if cache is not None and cache.is_file():
        payload = np.load(cache, allow_pickle=False)
        if payload["tag"].item() == f"{model}/sentence" and payload["n"].item() == len(captions):
            print(f"  特征缓存命中 {cache.name}")
            return payload["features"]
        if payload["tag"].item() == f"{model}/sentence":
            # 缓存存在但条数不同：说明数据集变了，重算（只有条数相同才能复用，否则对不上号）
            print(f"  ⚠️ {cache.name} 的条数与当前数据不一致，重算")

    from biosnn_bus.plugins.text_zscore import OllamaEmbedding

    backend = OllamaEmbedding(model=model)
    rows: list[np.ndarray] = []
    batch = 64
    for start in range(0, len(captions), batch):
        rows.extend(backend.embed_batch(captions[start : start + batch]))
        print(f"  nomic {min(start + batch, len(captions))}/{len(captions)}", end="\r")
    print()
    features = _l2(np.asarray(rows, dtype=np.float64))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache, features=features, n=np.array(len(captions)), tag=np.array(f"{model}/sentence")
        )
    return features


def describe(features: np.ndarray) -> str:
    """给复现记录用的一行摘要。"""
    return json.dumps({"n": int(features.shape[0]), "dim": int(features.shape[1])})

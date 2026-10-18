"""冻结前端：把 CIFAR 图像编码成特征（供持续学习基准使用）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 边界

仓库允许**冻结的预训练前端**（README「感知层允许冻结预训练前端」）：判据是**参数进不进梯度**，
不是模型大不大。这里 ResNet18 在 ``torch.no_grad()`` 下前向，权重不参与任何训练。

⚠️ **换前端会改变基准的含义**：W1 那条测的是"本仓的 Hebbian 表示"，
ResNet18 这条测的是"任何强表示 + 局部原型记忆"。两者要分别报，不能混为一谈。

## 为什么 CIFAR 要 resize 到 224

ImageNet 权重是在 224 上训的；直接在 32×32 上前向会得到分布外的激活。resize + 标准化是
**使用预训练权重的常规做法**，不是调参。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["IMAGE_NET_MEAN", "IMAGE_NET_STD", "resnet18_features"]

#: ImageNet 预训练权重的标准化常量。
IMAGE_NET_MEAN = (0.485, 0.456, 0.406)
IMAGE_NET_STD = (0.229, 0.224, 0.225)


def resnet18_features(
    images: np.ndarray,
    *,
    cache: Path | None = None,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """``(n, 32, 32, 3)`` uint8 图像 → ``(n, 512)`` 的 ResNet18（冻结）特征，**已 L2 归一化**。

    Args:
        images: CIFAR 形态的 uint8 图像。
        cache: ``.npz`` 缓存路径；命中时直接读盘（含条数与模型标识校验）。
        batch_size: 前向批量。
        device: ``"cuda"`` / ``"cpu"``；``None`` 时自动选。
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images 必须是 (n, H, W, 3)，收到 {images.shape}。")

    if cache is not None and cache.is_file():
        payload = np.load(cache, allow_pickle=False)
        if (
            payload["tag"].item() == "resnet18-imagenet1k-v1/224-cifar/avgpool"
            and payload["n"].item() == images.shape[0]
        ):
            print(f"  特征缓存命中 {cache.name}")
            return payload["features"]

    import torch
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ]
    )

    rows: list[np.ndarray] = []
    tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = preprocess(tensor[start : start + batch_size].float().div(255.0)).to(device)
            rows.append(model(batch).cpu().numpy().astype(np.float64))
    features = np.concatenate(rows, axis=0)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache,
            features=features,
            n=np.array(images.shape[0]),
            tag=np.array("resnet18-imagenet1k-v1/224-cifar/avgpool"),
        )
    return features

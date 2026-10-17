"""把配对集送进插件与总线，得到两种读法的表示。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

两种读法对应 `metrics.py` 文档里的 (a) 与 (b)：

* **(a) 编码质量**——插件**自身**的输出（图像 ``2·H·W``、文本 ``n_neurons``），
  只做**同模态**检索。它回答「编码有没有保住身份」，回答不了跨模态。
* **(b) 总线接受度**——``SpikeBus.step()`` 的输出，形状 ``(T, bus_dim)``，
  沿时间取均值后是 ``bus_dim`` 维。**这是验收线所在的表示。**

⚠️ **两路的 bus 表示不在同一空间**：``semantic`` 与 ``temporal`` 两个通道各自经过
独立种子的随机投影（见预注册）。所以 (b) 的跨模态检索**必须**经过一层联想，
直接把两路输出拿去比余弦只会得到随机水平——那是接口性质，不是编码缺陷。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from biosnn_bus import SpikeBus
from biosnn_bus.plugins import DiffImagePlugin, TextZScorePlugin
from biosnn_bus.plugins.text_zscore import EmbeddingBackend

from .procedural import CANVAS

__all__ = ["CachedEmbedding", "Representations", "build_bus", "encode_items"]


class CachedEmbedding(EmbeddingBackend):
    """给任意 :class:`EmbeddingBackend` 加一层磁盘缓存。

    **为什么必须缓存**：文本插件按 **token** 取 embedding，而一条 caption 有 9 个
    token、数据集有 144 条——重复调用 Ollama 会让一次运行从秒级变成分钟级。
    缓存的键是 token 原文，值是该 token 的向量；冻结前端意味着**同一个 token 永远
    对应同一个向量**，所以缓存不失真。

    ⚠️ 缓存只对**同一个模型**有效。换模型要换文件，否则会拿到另一个空间的向量
    而毫无提示——所以文件里存了模型名，不匹配就报错。
    """

    def __init__(self, inner: EmbeddingBackend, cache_path: Path | str, *, model: str) -> None:
        self.inner = inner
        self.path = Path(cache_path)
        self.model = model
        self._cache: dict[str, list[float]] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("model") != model:
                raise ValueError(
                    f"缓存模型 {payload.get('model')!r} 与当前 {model!r} 不一致——换个缓存文件。"
                )
            self._cache = payload.get("vectors", {})
        self._dirty = False

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = [float(x) for x in self.inner.embed(text)]
            self._dirty = True
        return np.asarray(self._cache[text], dtype=np.float64)

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model, "vectors": self._cache}
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._dirty = False


@dataclass(frozen=True)
class Representations:
    """一次编码的四种表示，行序与传入的 ``items`` 一致。"""

    text_plugin: np.ndarray
    image_plugin: np.ndarray
    text_bus: np.ndarray
    image_bus: np.ndarray

    @property
    def n(self) -> int:
        return self.text_bus.shape[0]


def build_bus(*, bus_dim: int, seed: int, backend: EmbeddingBackend | None = None) -> SpikeBus:
    """按预注册的固定配置建一台总线：图像 24×24、文本 256 群体、``bus_dim`` 维。"""
    bus = SpikeBus(bus_dim=bus_dim, seed=seed)
    bus.register(DiffImagePlugin(height=CANVAS, width=CANVAS))
    bus.register(TextZScorePlugin(backend=backend, seed=seed))
    return bus


def _mean_over_time(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float64).mean(axis=0)


def encode_items(bus: SpikeBus, items: list[Any]) -> Representations:
    """把每条样本分别以「只有文本」和「只有图像」送进总线。

    ⚠️ **分开送，不一起送**：一起送会得到两路之和，那是**配对**的表示，用来检索就等于
    把答案一起喂进去。检索要的是每一侧单独的表示。
    """
    image_plugin = bus.get("image_diff")
    text_plugin = bus.get("text_zscore")

    text_plugin_rows: list[np.ndarray] = []
    image_plugin_rows: list[np.ndarray] = []
    text_bus_rows: list[np.ndarray] = []
    image_bus_rows: list[np.ndarray] = []

    for item in items:
        text_train = text_plugin.encode(item.caption)
        image_train = image_plugin.encode(item.image)

        text_plugin_rows.append(_mean_over_time(text_train.data))
        image_plugin_rows.append(_mean_over_time(image_train.data))

        text_bus_rows.append(_mean_over_time(bus.step({"text_zscore": item.caption}).data))
        image_bus_rows.append(_mean_over_time(bus.step({"image_diff": item.image}).data))

    return Representations(
        text_plugin=np.stack(text_plugin_rows),
        image_plugin=np.stack(image_plugin_rows),
        text_bus=np.stack(text_bus_rows),
        image_bus=np.stack(image_bus_rows),
    )


def dump_config(**kwargs: Any) -> dict[str, Any]:
    """把这次编码的配置摊平成可写进复现记录的字典。"""
    return {
        k: (v if isinstance(v, (int, float, str, bool)) else repr(v)) for k, v in kwargs.items()
    }

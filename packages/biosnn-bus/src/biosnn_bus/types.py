"""骨架库的基础数据类型：脉冲序列与融合通道。

设计约束见 ``docs/adr/ADR-0002``：本模块只依赖 numpy。认知核心侧使用的 torch
张量通过 :meth:`SpikeTrain.to_torch` / :meth:`SpikeTrain.from_torch` 桥接，
骨架库不反向依赖 torch。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["FusionChannel", "Membrane", "SpikeTrain"]


class FusionChannel(str, Enum):
    """脉冲总线的融合通道。

    对应计划书 §2.2 的"双通道融合"与 §2.3 模态表中每一行的 ``融合通道`` 列。
    继承 :class:`str` 是为了让插件作者可以直接写 ``return "temporal"``，
    同时又能被 :class:`SpikeBus` 归一化成枚举。
    """

    TEMPORAL = "temporal"
    SEMANTIC = "semantic"

    def __str__(self) -> str:
        return self.value


@runtime_checkable
class Membrane(Protocol):
    """膜电位模块的最小结构契约。

    计划书 §2.3 把 ``get_membrane`` 的返回类型写作 ``nn.Module``。骨架库不依赖
    torch（ADR-0002），因此这里改为结构化协议：认知核心返回的 ``nn.Module``
    子类结构上自动满足本协议，插件签名语义不变。
    """

    def forward(self, x: Any) -> Any:
        """前向计算。"""
        ...


@dataclass(frozen=True, eq=False)
class SpikeTrain:
    """一段脉冲序列。

    形状约定为 ``(T, N)``：``T`` 是时间步数，``N`` 是神经元数。``data`` 可以是
    布尔数组（0/1 脉冲）或浮点数组（发放率 / 膜电位代理）。时间步 ``t`` 对应
    的物理时刻是 ``t * dt`` 毫秒。

    相等性按值比较（``data`` 逐元素 + 其余字段）。注意本类**按身份哈希**，不能
    用作字典键或放进 ``set``——numpy 数组无法给出稳定的值哈希。

    Attributes:
        data: ``(T, N)`` 的 numpy 数组。
        dt: 每个时间步的长度（毫秒），必须为正。
        temporal_scale: 该模态的典型时间尺度（毫秒），供 TAAF 模块初始化使用。
        channel: 该序列走哪条融合通道。
    """

    data: np.ndarray
    dt: float = 1.0
    temporal_scale: float = 10.0
    channel: FusionChannel = FusionChannel.TEMPORAL

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 2:
            raise ValueError(
                f"SpikeTrain.data 必须是 (T, N) 的二维数组，收到形状 {data.shape}。"
                f"若只有单个时间步，请用 data[None, :] 显式补一维。"
            )
        if data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError(f"SpikeTrain.data 的 T 与 N 都必须 >= 1，收到形状 {data.shape}。")
        if not np.issubdtype(data.dtype, np.bool_) and not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"SpikeTrain.data 的 dtype 必须是布尔或数值型，收到 {data.dtype}。")
        if np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
            raise ValueError("SpikeTrain.data 含 NaN 或 Inf。")

        dt = float(self.dt)
        if not dt > 0:
            raise ValueError(f"SpikeTrain.dt 必须为正，收到 {self.dt!r}。")
        temporal_scale = float(self.temporal_scale)
        if not temporal_scale > 0:
            raise ValueError(f"SpikeTrain.temporal_scale 必须为正，收到 {self.temporal_scale!r}。")

        channel = FusionChannel(self.channel)  # 允许传字符串

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "temporal_scale", temporal_scale)
        object.__setattr__(self, "channel", channel)

    # ---------------------------------------------------------------- 形状信息

    @property
    def n_steps(self) -> int:
        """时间步数 ``T``。"""
        return int(self.data.shape[0])

    @property
    def n_neurons(self) -> int:
        """神经元数 ``N``。"""
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        """序列总时长（毫秒）。"""
        return self.n_steps * self.dt

    @property
    def is_binary(self) -> bool:
        """``data`` 是否为 0/1 脉冲。"""
        return bool(np.issubdtype(self.data.dtype, np.bool_))

    @property
    def rates(self) -> np.ndarray:
        """每个神经元的平均发放率（单位：每步），形状 ``(N,)``。"""
        return self.data.astype(np.float64).mean(axis=0)

    @property
    def density(self) -> float:
        """非零元素占比，落在 ``[0, 1]``。对应计划书 §9 的"脉冲稀疏度"指标。"""
        return float(np.count_nonzero(self.data) / self.data.size)

    @property
    def spike_count(self) -> int:
        """脉冲总数。"""
        return int(np.count_nonzero(self.data))

    # ---------------------------------------------------------------- 变换

    def binary(self, threshold: float = 0.5) -> SpikeTrain:
        """阈值化。对已经是布尔的序列返回自身。"""
        if self.is_binary:
            return self
        data = np.asarray(self.data) > threshold
        return self.replace(data=data)

    def resample(self, n_steps: int) -> SpikeTrain:
        """重采样到 ``n_steps`` 个时间步。

        采用**分组取最大值**：每个目标时间步覆盖一段源时间步，取该段内的最大值。
        这保证了

        * 降采样（目标步数更少）时窗口内的脉冲不会被丢掉——这是脉冲数据该有的
          语义，用均值会把稀疏脉冲稀释成噪声；
        * 升采样（目标步数更多）时每个源时间步只映射到唯一的目标时间步，既不复制
          脉冲，也不会凭空造出新的脉冲。

        时间步长 ``dt`` 按 ``duration / n_steps`` 同步缩放，因此总时长保持不变。
        """
        new_t = int(n_steps)
        if new_t < 1:
            raise ValueError(f"n_steps 必须 >= 1，收到 {n_steps!r}。")
        if new_t == self.n_steps:
            return self

        src_t = self.n_steps
        # edges[i] = 目标时间步 i 在源时间轴上的起点（整数，可能重复）
        edges = np.linspace(0, src_t, new_t + 1).astype(np.intp)
        # group[t] = 源时间步 t 归属的目标时间步
        group = np.searchsorted(edges[1:], np.arange(src_t), side="right")
        np.clip(group, 0, new_t - 1, out=group)

        source = self.data
        was_bool = self.is_binary
        work = source.astype(np.uint8) if was_bool else source
        out = np.zeros((new_t, self.n_neurons), dtype=work.dtype)
        np.maximum.at(out, group, work)
        if was_bool:
            out = out.astype(bool)

        return self.replace(data=out, dt=self.duration / new_t)

    def replace(self, **changes: Any) -> SpikeTrain:
        """返回替换了部分字段的新 :class:`SpikeTrain`。"""
        fields: dict[str, Any] = {
            "data": self.data,
            "dt": self.dt,
            "temporal_scale": self.temporal_scale,
            "channel": self.channel,
        }
        unknown = set(changes) - set(fields)
        if unknown:
            raise TypeError(f"SpikeTrain 没有这些字段：{sorted(unknown)}")
        fields.update(changes)
        return SpikeTrain(**fields)

    # ---------------------------------------------------------------- torch 桥接

    def to_torch(self, dtype: Any = None, device: Any = None) -> Any:
        """转为 ``torch.Tensor``，形状仍为 ``(T, N)``。

        torch 是可选依赖：未安装时抛 :class:`ImportError` 并提示安装 extra。
        """
        torch = _import_torch()
        tensor = torch.from_numpy(np.ascontiguousarray(self.data))
        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @classmethod
    def from_torch(
        cls,
        tensor: Any,
        *,
        dt: float = 1.0,
        temporal_scale: float = 10.0,
        channel: str | FusionChannel = FusionChannel.TEMPORAL,
    ) -> SpikeTrain:
        """从 ``torch.Tensor`` 构造。会自动 ``detach`` 并搬到 CPU。

        返回的 ``SpikeTrain`` **不与入参张量共享内存**。这一点必须显式保证：张量本来
        就在 CPU 上时，``.to("cpu")`` 是空操作，紧随其后的 ``.numpy()`` 会返回一个
        与张量共享缓冲区的视图——于是 ``train.data[0, 0] = 1`` 会静默改写调用方的
        张量，而 ``SpikeTrain`` 是**按值**比较的不可变数据类型，共享缓冲区与它的值
        语义直接矛盾。
        """
        _import_torch()
        array = tensor.detach().to("cpu").numpy()
        if array.base is not None:
            # 与入参张量（或它的某个视图）共享内存，复制出来切断这层关系。
            # 只在真的共享时才复制——GPU 张量经 .to("cpu") 已经拿到独立缓冲，
            # 那一路上 .numpy() 的 base 是那个新建的 CPU 张量，会走到这里多复制一次，
            # 但这条路径本来就绕不开一次拷贝，代价可以接受。
            array = array.copy()
        return cls(data=array, dt=dt, temporal_scale=temporal_scale, channel=channel)

    # ---------------------------------------------------------------- 展示

    def __eq__(self, other: object) -> bool:
        """按值相等：``data`` 逐元素比较，其余字段精确比较。

        之所以手写而不用 dataclass 自动生成的 ``__eq__``：后者会把字段放进元组
        比较，遇到 numpy 数组会返回数组而不是布尔值，``train1 == train2`` 直接抛
        ``ValueError: The truth value of an array ... is ambiguous``。
        """
        if not isinstance(other, SpikeTrain):
            return NotImplemented
        return (
            self.channel is other.channel
            and self.dt == other.dt
            and self.temporal_scale == other.temporal_scale
            and self.data.shape == other.data.shape
            and bool(np.array_equal(self.data, other.data))
        )

    def __repr__(self) -> str:
        kind = "binary" if self.is_binary else str(self.data.dtype)
        return (
            f"SpikeTrain(shape=({self.n_steps}, {self.n_neurons}), {kind}, "
            f"dt={self.dt:g}ms, duration={self.duration:g}ms, "
            f"channel={self.channel.value}, density={self.density:.3g})"
        )


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "该功能需要 PyTorch。请安装可选依赖：pip install 'biosnn-bus[torch]'"
        ) from exc
    return torch

"""脉冲总线：把多个模态插件的脉冲序列对齐、融合、投影到认知核心的输入维度。

.. warning::

   本模块目前是**骨架**。计划书 §2.2 描述的真实脉冲总线集成了 **TAAF + 双通道
   融合** [Shen et al., 2025]，其中 TAAF 模块"在每个时间步动态分配重要性分数，
   实现时间异构脉冲特征的分层整合"。那属于计划书 §七 **第二阶段**的研究任务，
   第零阶段只搭骨架与接缝：

   * :class:`FusionStrategy` 是 TAAF 未来插入的位置——换策略不改调用方；
   * :class:`SpikeBus` 已经把输入按 :class:`~biosnn_bus.types.FusionChannel`
     分成"时间""语义"两条通路，对应"双通道融合"；
   * 默认策略 :class:`ConcatenateFusion` 是平凡实现，**不是** TAAF。

   骨架库对外的接口稳定性承诺见 ``docs/adr/ADR-0005``：``biosnn-bus`` 遵循
   semver，替换默认融合策略不算破坏性变更。
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .errors import (
    BioSNNBusError,
    PluginContractError,
    UnknownModalityError,
)
from .plugin import ModalityPlugin
from .types import FusionChannel, SpikeTrain

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "ConcatenateFusion",
    "FusionStrategy",
    "SparseRandomProjection",
    "SpikeBus",
]


@runtime_checkable
class FusionStrategy(Protocol):
    """把同一融合通道内的多路脉冲序列合并成一路。

    实现者只需保证：输入的所有序列已经**对齐到同一时间网格**（相同 ``T``、相同
    ``dt``），且属于同一 :class:`~biosnn_bus.types.FusionChannel`。
    """

    def fuse(self, trains: Sequence[SpikeTrain]) -> SpikeTrain:
        """融合。``trains`` 非空。"""
        ...


class ConcatenateFusion:
    """平凡融合策略：沿神经元维拼接。

    这是骨架库的默认策略，用来说明接口怎么接，**不是**计划书 §2.2 的 TAAF。
    融合后的 ``temporal_scale`` 取各输入的最大值（最慢的模态决定融合后的尺度）。
    """

    def fuse(self, trains: Sequence[SpikeTrain]) -> SpikeTrain:
        if not trains:
            raise BioSNNBusError("ConcatenateFusion 至少需要一路输入。")
        channels = {t.channel for t in trains}
        if len(channels) > 1:
            raise BioSNNBusError(
                f"融合前必须先按通道分组，收到混合通道 {sorted(c.value for c in channels)}。"
            )

        data = np.concatenate([t.data for t in trains], axis=1)
        return SpikeTrain(
            data=data,
            dt=trains[0].dt,
            temporal_scale=max(t.temporal_scale for t in trains),
            channel=trains[0].channel,
        )


class SparseRandomProjection:
    """固定种子的稀疏随机投影：``(T, in_dim) -> (T, out_dim)``。

    权重取 ``±1``，连接率由 ``density`` 控制。±1 的随机投影近似保距离
    （Johnson–Lindenstrauss），这正是认知核心需要的性质：能被区分开的感知模式，
    投影之后仍然能被区分开。

    每个输入维度固定连出 ``round(out_dim * density)`` 条边（至少 1 条），而不是在
    整个矩阵上均匀撒点。这样保证**没有输入维度会被随机丢空**——否则一个恰好在
    投影里没有出边的输入神经元，它的活动对认知核心完全不可见，而且哪个神经元中招
    取决于种子，排查起来极其困难。

    投影矩阵以 COO 三元组存储，因此 ``in_dim`` 或 ``out_dim`` 很大时也不会撑爆
    内存；代价是 :func:`numpy.add.at` 的散加比稠密矩阵乘法慢。骨架阶段够用，
    如果后续成为瓶颈，替换实现即可（对外接口不变）。
    """

    def __init__(self, in_dim: int, out_dim: int, *, density: float = 0.1, seed: int = 0) -> None:
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim 与 out_dim 都必须为正，收到 {in_dim} 与 {out_dim}。")
        if not 0 < density <= 1:
            raise ValueError(f"density 必须落在 (0, 1]，收到 {density}。")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.density = float(density)

        rng = np.random.default_rng(seed)
        n_per_input = max(1, round(self.out_dim * self.density))
        n_conn = self.in_dim * n_per_input
        self._cols = np.repeat(np.arange(self.in_dim, dtype=np.intp), n_per_input)
        self._rows = rng.integers(0, self.out_dim, size=n_conn).astype(np.intp)
        self._signs = rng.choice(np.array([-1.0, 1.0]), size=n_conn)

    @property
    def n_connections(self) -> int:
        """非零权重个数。"""
        return int(self._rows.size)

    @property
    def fan_out_per_input(self) -> int:
        """每个输入维度连出的边数。"""
        return self.n_connections // self.in_dim

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """应用投影。``data`` 形状 ``(T, in_dim)``，返回 ``(T, out_dim)`` 的 float32。"""
        array = np.asarray(data)
        if array.ndim != 2 or array.shape[1] != self.in_dim:
            raise ValueError(f"投影期望形状 (T, {self.in_dim}) 的输入，收到 {array.shape}。")
        contrib = array[:, self._cols].astype(np.float32) * self._signs
        out = np.zeros((array.shape[0], self.out_dim), dtype=np.float32)
        np.add.at(out, (slice(None), self._rows), contrib)
        return out

    def __repr__(self) -> str:
        return (
            f"SparseRandomProjection({self.in_dim} -> {self.out_dim}, "
            f"density={self.density:g}, n_connections={self.n_connections})"
        )


class SpikeBus:
    """模态插件与认知核心之间的脉冲总线。

    Args:
        bus_dim: 认知核心入口的神经元数。所有模态的输出最终都投影到这个维度。
        dt: 总线的时间步长（毫秒）。各模态先按各自 ``dt`` 编码，再重采样到本步长。
        strategy: 融合策略。``None`` 时用 :class:`ConcatenateFusion`。
        projection_density: 投影矩阵连接率。
        seed: 投影矩阵的随机种子。固定种子是复现的前提（见
            ``docs/reproducibility.md``）。

    Example:
        >>> from biosnn_bus import SpikeBus
        >>> from biosnn_bus.plugins import DiffImagePlugin
        >>> bus = SpikeBus(bus_dim=128, seed=0)
        >>> bus.register(DiffImagePlugin(height=8, width=8))
        >>> frames = np.zeros((5, 8, 8), dtype=np.float32)
        >>> out = bus.step({"image_diff": frames})  # doctest: +SKIP
    """

    def __init__(
        self,
        bus_dim: int,
        *,
        dt: float = 1.0,
        strategy: FusionStrategy | None = None,
        projection_density: float = 0.1,
        seed: int = 0,
    ) -> None:
        if isinstance(bus_dim, bool) or not isinstance(bus_dim, int) or bus_dim <= 0:
            raise ValueError(f"bus_dim 必须是正整数，收到 {bus_dim!r}。")
        if not dt > 0:
            raise ValueError(f"dt 必须为正，收到 {dt!r}。")

        self.bus_dim = int(bus_dim)
        self.dt = float(dt)
        self.strategy: FusionStrategy = strategy or ConcatenateFusion()
        self.projection_density = float(projection_density)
        self.seed = int(seed)

        self._plugins: dict[str, ModalityPlugin] = {}
        self._projections: dict[tuple[Any, ...], SparseRandomProjection] = {}

    # ------------------------------------------------------------ 注册

    @property
    def registered_modalities(self) -> tuple[str, ...]:
        """已注册的模态名（按注册顺序）。"""
        return tuple(self._plugins)

    def register(self, plugin: ModalityPlugin, *, name: str | None = None) -> str:
        """注册一个模态插件。

        Args:
            plugin: :class:`~biosnn_bus.plugin.ModalityPlugin` 实例。
            name: 覆盖插件自报的 ``modality_name``，用于同一模态注册多份配置
                （例如两个不同分辨率的图像编码器）。

        Returns:
            实际使用的注册名。

        Raises:
            PluginContractError: 插件不满足接口契约，或在同一总线上重名。
        """
        if not isinstance(plugin, ModalityPlugin):
            raise PluginContractError(
                f"只能注册 ModalityPlugin 的实例，收到 {type(plugin).__name__}。"
            )
        plugin.validate()

        key = name if name is not None else plugin.modality_name
        if not isinstance(key, str) or not key.strip():
            raise PluginContractError(f"注册名必须是非空字符串，收到 {key!r}。")
        if key in self._plugins:
            raise PluginContractError(
                f"本总线上模态名 {key!r} 已被占用。若同一模态要注册多份配置，请显式传 name=。"
            )

        self._plugins[key] = plugin
        return key

    def unregister(self, name: str) -> None:
        """注销一个模态。"""
        if name not in self._plugins:
            self._raise_unknown(name)
        del self._plugins[name]

    def get(self, name: str) -> ModalityPlugin:
        """按注册名取回插件实例。

        Raises:
            UnknownModalityError: 未注册该名称。
        """
        plugin = self._plugins.get(name)
        if plugin is None:
            self._raise_unknown(name)
        return plugin

    # ------------------------------------------------------------ 前向

    def step(self, inputs: Mapping[str, Any]) -> SpikeTrain:
        """编码并融合一批模态输入。

        Args:
            inputs: ``{模态名: 原始输入}``。键必须在已注册模态中。

        Returns:
            形状 ``(T, bus_dim)`` 的 :class:`~biosnn_bus.types.SpikeTrain`。两点注意：

            * ``data`` 是**浮点**的，表示融合后送进认知核心的输入电流（稀疏随机
              投影带 ±1 权重），**不是** 0/1 脉冲。需要二值化时调用 ``.binary()``；
            * ``channel`` 字段在融合之后已无意义（输出是双通道之和），保留默认值
              仅为满足容器契约。

        Raises:
            ValueError: ``inputs`` 为空。
            UnknownModalityError: 某个键未注册。
            PluginContractError: ``encode`` 的返回值类型或维度不符合 ``spike_dim``。
        """
        if not inputs:
            raise ValueError("step() 至少需要一路输入。")

        pairs: list[tuple[str, SpikeTrain]] = []
        for modality, raw in inputs.items():
            plugin = self._plugins.get(modality)
            if plugin is None:
                self._raise_unknown(modality)
                raise AssertionError("unreachable")  # pragma: no cover
            pairs.append((modality, self._encode(plugin, modality, raw)))

        aligned = self._align([train for _, train in pairs])
        pairs = [(name, train) for (name, _), train in zip(pairs, aligned, strict=True)]

        fused: list[np.ndarray] = []
        for channel in FusionChannel:
            # 按模态名排序 —— 融合是沿神经元维拼接，列序会改变投影矩阵的含义，
            # 因此必须用与输入字典顺序无关的规范顺序，否则同一组输入换个传入顺序
            # 就会得到不同的总线输出，破坏可复现性。
            group = sorted(
                ((name, train) for name, train in pairs if train.channel is channel),
                key=lambda item: item[0],
            )
            if not group:
                continue
            merged = self.strategy.fuse([train for _, train in group])
            projection = self._get_projection(channel, group)
            fused.append(projection(merged.data))

        if not fused:  # pragma: no cover - 输入非空时不可能走到
            raise BioSNNBusError("没有可融合的脉冲序列。")

        stacked = np.sum(fused, axis=0) if len(fused) > 1 else fused[0]
        return SpikeTrain(
            data=stacked,
            dt=self.dt,
            temporal_scale=max(train.temporal_scale for _, train in pairs),
        )

    # ------------------------------------------------------------ 内部

    def _encode(self, plugin: ModalityPlugin, modality: str, raw: Any) -> SpikeTrain:
        train = plugin.encode(raw)
        if not isinstance(train, SpikeTrain):
            raise PluginContractError(
                f"插件 {modality!r}（{type(plugin).__name__}）的 encode() 返回了 "
                f"{type(train).__name__}，必须是 SpikeTrain。"
            )
        expected = plugin.spike_dim
        if train.n_neurons != expected:
            raise PluginContractError(
                f"插件 {modality!r} 声明 spike_dim={expected}，但 encode() 返回了 "
                f"{train.n_neurons} 个神经元（形状 {train.data.shape}）。"
            )
        # 以插件声明的 fusion_channel 为准：encode() 里忘了设 channel 也能正常工作。
        return train.replace(channel=plugin.channel, temporal_scale=plugin.temporal_scale)

    def _align(self, trains: Sequence[SpikeTrain]) -> list[SpikeTrain]:
        """把各模态重采样到统一的 ``dt`` 与统一的时间步数。"""
        duration = max(t.duration for t in trains)
        n_steps = max(1, round(duration / self.dt))
        return [t.resample(n_steps) for t in trains]

    def _get_projection(
        self, channel: FusionChannel, group: Sequence[tuple[str, SpikeTrain]]
    ) -> SparseRandomProjection:
        in_dim = sum(train.n_neurons for _, train in group)
        # 缓存键含模态名与各自维度，且 group 已按模态名排序，因此与本步传入的
        # 输入字典顺序无关。注意用 zlib.crc32 而不是内置 hash()——后者对 str 是
        # 按进程加盐的，会让同一份配置在不同进程里得到不同的投影矩阵，
        # 破坏可复现性（见 docs/reproducibility.md）。
        key = (channel.value, tuple((name, train.n_neurons) for name, train in group))
        projection = self._projections.get(key)
        if projection is None:
            projection = SparseRandomProjection(
                in_dim,
                self.bus_dim,
                density=self.projection_density,
                seed=self._derive_seed(key),
            )
            self._projections[key] = projection
        return projection

    def _derive_seed(self, key: tuple[Any, ...]) -> int:
        tag = "|".join([str(self.seed), *[str(part) for part in key]]).encode()
        return zlib.crc32(tag)

    def _raise_unknown(self, modality: Any) -> None:
        known = ", ".join(sorted(self._plugins)) or "（空）"
        raise UnknownModalityError(
            f"未在本总线注册的模态 {modality!r}。已注册：{known}。"
            f"（注意：register_plugin 只写入全局注册表，还要 bus.register(plugin) 才会生效。）"
        )

    # ------------------------------------------------------------ 自省

    def describe(self) -> dict[str, Any]:
        """返回总线当前状态的摘要，便于打日志与写实验记录。"""
        by_channel: dict[str, list[str]] = {c.value: [] for c in FusionChannel}
        for name, plugin in self._plugins.items():
            by_channel[plugin.channel.value].append(name)
        return {
            "bus_dim": self.bus_dim,
            "dt": self.dt,
            "seed": self.seed,
            "projection_density": self.projection_density,
            "strategy": type(self.strategy).__name__,
            "modalities": {
                name: {
                    "plugin": type(plugin).__name__,
                    "spike_dim": plugin.spike_dim,
                    "channel": plugin.channel.value,
                    "temporal_scale": plugin.temporal_scale,
                }
                for name, plugin in self._plugins.items()
            },
            "modalities_by_channel": by_channel,
        }

    def __repr__(self) -> str:
        names = ", ".join(self._plugins) or "-"
        return (
            f"SpikeBus(bus_dim={self.bus_dim}, dt={self.dt:g}ms, "
            f"strategy={type(self.strategy).__name__}, modalities=[{names}])"
        )

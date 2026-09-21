# 插件开发指南

> 本文档里的每个 Python 代码块都会被 CI **真正执行**（见
> `scripts/check_doc_code_blocks.py`）。你可以从头到尾复制粘贴，它们是连着的。
> 无法独立运行的片段必须在围栏上标注 `python no-run`。

## 你要写的是什么

一个模态插件回答三个问题：

| 方法 | 回答的问题 |
| :--- | :--- |
| `encode(raw_input)` | 我的模态数据怎么变成脉冲？ |
| `get_membrane()` | 我这个模态的神经元长什么样？ |
| `decode(spike_output)` | 脉冲怎么变回我能看懂的东西？ |

外加两条元信息：`modality_name`（叫什么）和 `spike_dim`（输出多少路）。

这就是全部。你不用管学习规则、不用管总线怎么融合、不用管认知核心是什么——
那些是骨架库和认知核心的事。

## 一个完整的最小插件

下面这段是可以直接跑的。它实现一个**群体水平编码**：把一维信号编码成一组神经元的
发放，信号越接近哪个神经元偏好的水平，哪个神经元就越可能放电。

```python
import numpy as np

from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeTrain


class LevelEncoder(ModalityPlugin):
    """把一维信号按水平编码成群体脉冲。"""

    def __init__(self, spike_dim: int = 16) -> None:
        if spike_dim < 2:
            raise ValueError("spike_dim 至少为 2，否则无法区分信号水平。")
        self._spike_dim = spike_dim
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    # ---- 两个属性 ----

    @property
    def modality_name(self) -> str:
        return "level"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    # ---- 三个方法 ----

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        radius = 0.5 / (self._spike_dim - 1)  # 相邻水平的半间距
        distance = np.abs(signal[:, None] - self.levels[None, :])
        return SpikeTrain(data=distance <= radius, channel=self.channel)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output: SpikeTrain):
        active = spike_output.data.astype(np.float64)
        total = active.sum(axis=1)
        return np.divide(
            active @ self.levels, total, out=np.zeros(active.shape[0]), where=total > 0
        )
```

试一下：

```python
plugin = LevelEncoder(spike_dim=16)
signal = np.linspace(0.0, 1.0, 8)

train = plugin.encode(signal)
print(train)  # 形状 (8, 16)、布尔、channel=temporal
print("解码回来:", np.round(plugin.decode(train), 3))
```

### `spike_dim` 必须与 `encode` 的实际输出一致

总线会检查这一点。声明 16 却返回 32 路，注册时不会报错，但第一次 `step()` 就会被拦下：

```python
from biosnn_bus import PluginContractError, SpikeBus


class WrongDim(LevelEncoder):
    @property
    def spike_dim(self) -> int:
        return 16

    def encode(self, raw_input) -> SpikeTrain:
        return SpikeTrain(data=np.zeros((4, 99), dtype=bool))


bus = SpikeBus(bus_dim=32, seed=0)
bus.register(WrongDim())  # 注册时不查，因为还没调用过 encode
try:
    bus.step({"level": signal})
except PluginContractError as exc:
    print("如期被拦下:", exc)
```

> 注册时查不了声明与输出是否一致：`spike_dim` 是静态声明，`encode` 的输出是运行时
> 行为。所以在 `encode` 里就用 `self.spike_dim` 构造输出，两者不会错位
> （上面的 `LevelEncoder.encode` 就是这么写的）。为什么校验不放在 `__init_subclass__`，
> 见 [ADR-0006](../docs/adr/ADR-0006-plugin-interface-fidelity.md)。

## 两条可选属性

### `fusion_channel`：走哪条融合通道

计划书 §2.2 的脉冲总线是**双通道**的。默认 `'temporal'`，另一个取值是 `'semantic'`。
选哪个取决于**这个模态的信息以什么方式组织**：

| 取值 | 适合 | 计划书 §2.3 里的例子 |
| :--- | :--- | :--- |
| `'temporal'` | 信息主要在时间结构里（节奏、次序、时长） | 文本、音频 |
| `'semantic'` | 信息主要在内容/空间结构里 | 图像 |

```python
class SemanticEncoder(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "semantic"


print(SemanticEncoder().channel)  # 打印为 "semantic"
```

写错会怎样：

```python
class BadChannel(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "auditory"  # 没有这个通道


try:
    SpikeBus(bus_dim=32).register(BadChannel())
except PluginContractError as exc:
    print("如期被拦下:", exc)
```

### `temporal_scale`：这个模态的时间尺度

单位是毫秒，默认 `10.0`。计划书把它定义为"该模态的典型时间尺度，供 TAAF 模块初始化
使用"。骨架阶段它作为一个标记随脉冲序列传递，融合时取各模态的**最大值**
（最慢的模态决定融合后的时间尺度）。

```python
class FastEncoder(LevelEncoder):
    @property
    def temporal_scale(self) -> float:
        return 1.0


print(FastEncoder().temporal_scale)
```

## 契约自检

`SpikeBus.register()` 会调用插件的 `validate()`，以下四类问题在那一刻被拦下：

```python
class BadName(LevelEncoder):
    @property
    def modality_name(self) -> str:
        return "   "


class BadDim(LevelEncoder):
    @property
    def spike_dim(self) -> int:
        return -3


class BadScale(LevelEncoder):
    @property
    def temporal_scale(self) -> float:
        return 0.0


class BadChannel(LevelEncoder):
    @property
    def fusion_channel(self) -> str:
        return "auditory"


for bad in (BadName(), BadDim(), BadScale(), BadChannel()):
    try:
        SpikeBus(bus_dim=32).register(bad)
    except PluginContractError as exc:
        print(f"{type(bad).__name__}: {exc}")
```

## 接进总线

```python
bus = SpikeBus(bus_dim=64, seed=0)
bus.register(LevelEncoder(spike_dim=16))

output = bus.step({"level": signal})
print(output)  # 形状 (8, 64)，float32
```

注意两点：

1. `output.data` 是**浮点电流**，不是 0/1 脉冲。总线末端的稀疏随机投影带 ±1 权重，
   负值表示抑制性输入。需要脉冲时调用 `output.binary()`。
2. 同一个总线上的模态名必须唯一。同一个模态要注册多份配置（比如两个分辨率的图像
   编码器），用 `name=` 显式区分：

```python
bus.register(LevelEncoder(spike_dim=16), name="level_low")
bus.register(LevelEncoder(spike_dim=64), name="level_high")
print(bus.registered_modalities)
```

## 让第三方包提供插件

**你不需要修改本仓库任何一行代码。** 在你的包的 `pyproject.toml` 里声明：

```toml
[project.entry-points."biosnn_bus.plugins"]
audio = "my_pkg.plugins.audio:AudioPlugin"
```

用户装完你的包后：

```python no-run
# 这段依赖一个真实存在的第三方包，本仓库里当然没有装，所以无法执行。
from biosnn_bus import SpikeBus, discover_plugins, get_plugin

discover_plugins()  # 扫描并注册所有已安装分发包声明的插件
bus = SpikeBus(bus_dim=256, seed=0)
bus.register(get_plugin("audio")())
```

`discover_plugins()` 是幂等的，重复调用不会重复注册。若某个 entry point 指向的不是
`ModalityPlugin` 子类，或者加载时抛 ImportError，错误信息里会带上 entry point 的名字
和它的导入路径。

## 测试模板

插件是该测的：契约违约要在**注册时**就暴露，而不是训练到一半才发现。这是本仓库
自带示例插件 `DiffImagePlugin` 的测试结构，可以直接照搬
（见 `packages/biosnn-bus/tests/test_plugins_image_diff.py`）：

```python
import numpy as np
import pytest

from biosnn_bus import SpikeBus, SpikeTrain
from biosnn_bus.plugins import DiffImagePlugin


@pytest.fixture
def plugin():
    return DiffImagePlugin(height=4, width=5, threshold=0.25)


def test_shape_contract(plugin):
    train = plugin.encode(np.zeros((6, 4, 5)))
    assert train.data.shape == (6, plugin.spike_dim)
    assert train.is_binary


def test_rejects_wrong_shape(plugin):
    with pytest.raises(ValueError, match="期望形状"):
        plugin.encode(np.zeros((6, 4, 6)))


def test_survives_a_round_trip(plugin):
    """编码可以有损，但解回来的形状必须对，且误差有明确上界。"""
    frames = np.zeros((4, 4, 5))
    frames[1, 0, 0] = 0.25  # 恰好等于阈值，应当触发一个事件
    restored = plugin.decode(plugin.encode(frames))
    assert restored.shape == frames.shape
    assert np.abs(restored - frames).max() <= 0.25


def test_plugs_into_the_bus(plugin):
    bus = SpikeBus(bus_dim=32, seed=0)
    bus.register(plugin)
    assert bus.step({"image_diff": np.zeros((6, 4, 5))}).data.shape == (6, 32)
```

**形状契约与拒绝错误形状这两条是必须的**——它们把"插件写错了"从训练中途才暴露，
变成注册时一行报错。

## 再往下

- 插件接口的完整签名与语义：`packages/biosnn-bus/src/biosnn_bus/plugin.py`
- 脉冲序列容器 `SpikeTrain` 的完整 API：`packages/biosnn-bus/src/biosnn_bus/types.py`
- 为什么要用 entry points 而不是全局注册表：`docs/adr/ADR-0003-entry-point-plugin-discovery.md`
- 骨架库**不**包含什么（以及为什么）：`docs/adr/ADR-0001-skeleton-as-separate-library.md`

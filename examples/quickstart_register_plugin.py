# %% [markdown]
# # 5 分钟：注册一个自定义模态插件
#
# 这个例子演示 `biosnn-bus` 要解决的唯一问题：**新增模态不需要修改已有架构**。
#
# 依次做四件事：写一个全新的模态插件（把一维信号编码成群体脉冲）、注册它、
# 和自带的图像插件一起接进 `SpikeBus`、看双通道路由与脉冲栅格图。
#
# 全程纯 CPU，唯一依赖是 numpy（绘图用 matplotlib）。

# %%
# **在 Colab 这类全新环境里，这个包还没有被安装。** 本地开发时它已经在环境里，
# 所以下面这段会直接跳过——只有 import 真的失败时才会去装。
import subprocess
import sys

try:
    import biosnn_bus
except ModuleNotFoundError:
    print("环境里没有 biosnn-bus，先从 PyPI 安装……")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "biosnn-bus"])
    import biosnn_bus  # noqa: F401

# %%
import numpy as np
from biosnn_bus import (
    ModalityPlugin,
    PassThroughMembrane,
    SpikeBus,
    SpikeTrain,
    register_plugin,
)
from biosnn_bus.plugins import DiffImagePlugin

# %% [markdown]
# ## 1. 定义插件
#
# 实现计划书 §2.3 规定的接口：`encode` / `get_membrane` / `decode` 三个方法，
# 加上 `modality_name` / `spike_dim` 两个属性。`fusion_channel` 与
# `temporal_scale` 有默认值，按需覆盖。
#
# 这里用的是**群体水平编码**：`spike_dim` 个神经元各自"偏好"一个信号水平，信号
# 落在谁的地盘上谁就放电，正好用来看清总线在做什么。


# %%
# override=True 是为了让这个单元可以反复重跑（Colab 里很常见）。
@register_plugin("sine_wave", override=True)
class SineWavePlugin(ModalityPlugin):
    """把一维信号按水平编码成群体脉冲（自定义模态示例）。"""

    def __init__(self, spike_dim: int = 32) -> None:
        if spike_dim < 2:
            raise ValueError("spike_dim 至少为 2，否则无法区分信号水平。")
        self._spike_dim = spike_dim
        # 每个神经元偏好的信号水平，均匀铺满 [0, 1]
        self.levels = np.linspace(0.0, 1.0, spike_dim)

    @property
    def modality_name(self) -> str:
        return "sine_wave"

    @property
    def spike_dim(self) -> int:
        return self._spike_dim

    @property
    def fusion_channel(self) -> str:
        # 时间序列走时间通道；图像插件走语义通道。总线的双通道路由就是靠这个属性。
        return "temporal"

    @property
    def temporal_scale(self) -> float:
        return 5.0

    def encode(self, raw_input) -> SpikeTrain:
        signal = np.asarray(raw_input, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(signal)):
            raise ValueError("输入信号含 NaN 或 Inf。")
        # 每个时间步，最接近信号水平的那个（或相邻两个）神经元放电。
        # 半径取相邻水平的半间距，保证任何信号值都至少有一个神经元响应。
        radius = 0.5 / (self._spike_dim - 1)
        distance = np.abs(signal[:, None] - self.levels[None, :])
        data = distance <= radius
        return SpikeTrain(data=data, channel=self.channel, temporal_scale=self.temporal_scale)

    def decode(self, spike_output: SpikeTrain):
        active = spike_output.data.astype(np.float64)
        total = active.sum(axis=1)
        # 按活跃神经元偏好的水平取加权平均；整步静默时记 0
        return np.divide(
            active @ self.levels,
            total,
            out=np.zeros(active.shape[0]),
            where=total > 0,
        )

    def get_membrane(self) -> PassThroughMembrane:
        # 骨架库不含学习规则；认知核心侧会在这一步接上真正的脉冲神经元层。
        return PassThroughMembrane()


# %% [markdown]
# ## 2. 接进脉冲总线
#
# `SpikeBus` 接收任意数量的插件，按各自的 `fusion_channel` 分成时间/语义两路，
# 对齐到同一时间网格，融合后投影到认知核心的输入维度。

# %%
bus = SpikeBus(bus_dim=128, seed=0)
bus.register(SineWavePlugin(spike_dim=32))
bus.register(DiffImagePlugin(height=8, width=8, threshold=0.1))

print(bus)
for name, info in bus.describe()["modalities"].items():
    print(f"  {name:12s} dim={info['spike_dim']:4d} channel={info['channel']}")
print("双通道路由:", bus.describe()["modalities_by_channel"])

# %% [markdown]
# ## 3. 跑一步
#
# 时间通道收一维正弦信号，语义通道收一段 8×8 的亮度变化序列。

# %%
steps = 24
t = np.linspace(0, 2 * np.pi, steps)
signal = 0.5 + 0.45 * np.sin(t)  # 落在 [0.05, 0.95]

rng = np.random.default_rng(0)
frames = np.zeros((steps, 8, 8))
frames[:, 2:6, 2:6] = 0.8  # 中间一块亮度
frames[12:, 0:2, 0:2] = 0.6  # 后半段左上角亮起

output = bus.step({"sine_wave": signal, "image_diff": frames})
print("总线输出:", output)
print(
    f"零值占比 {(output.data == 0).mean():.1%}（随机投影带 ±1 权重，输出是输入电流，不是 0/1 脉冲）"
)

# %% [markdown]
# ## 4. 看脉冲栅格图

# %%

import matplotlib.pyplot as plt  # noqa: E402

sine_plugin = bus.get("sine_wave")
image_plugin = bus.get("image_diff")

# 取"正电流"的单元——即真正被推高的那些认知核心神经元（见上一节说明）。
output_units = output.binary(threshold=0.0)

# 图里的标签用英文：默认的 DejaVu Sans 没有中文字形，中文标签在多数环境
# （含 Colab）会渲染成方块。正文说明仍然用中文。
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

for ax, train, title in (
    (axes[0], sine_plugin.encode(signal), "temporal channel - sine_wave (32 units, level code)"),
    (axes[1], image_plugin.encode(frames), "semantic channel - image_diff (128 ch, ON/OFF)"),
    (
        axes[2],
        output_units,
        f"cognitive core input - SpikeBus out ({bus.bus_dim}-d, fused, positive current only)",
    ),
):
    rows, cols = np.nonzero(train.data)
    ax.scatter(cols, rows, s=6, marker="|", linewidths=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("time step")
    ax.set_xlim(-0.5, train.n_neurons - 0.5)

axes[-1].set_xlabel("unit index")
fig.suptitle("biosnn-bus: plugin -> channel -> fusion -> cognitive core input", fontsize=11)
fig.tight_layout()

# 无头后端（CI）调用 show() 只会刷一堆告警；交互式后端（含 Colab inline）才需要它
if plt.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "template", "cairo"}:
    plt.show()

# 论文用的 F1。**只在设了 `BIOSNN_FIGURE_DIR` 时才落盘**——这样 CI（不设）行为不变，
# 而 `scripts/make_figures.py` 能把同一份图稳定地重画出来。图的**内容只有这一份真相源**，
# 不另抄一遍到绘图脚本里（那样迟早会漂移）。
import os  # noqa: E402
from pathlib import Path  # noqa: E402

_figure_dir = os.environ.get("BIOSNN_FIGURE_DIR")
if _figure_dir:
    _out = Path(_figure_dir) / "f1-spike-raster.png"
    _out.parent.mkdir(parents=True, exist_ok=True)
    # `metadata={"Software": None}` 去掉 PNG 里会变的软件版本串，让产物可逐字节比对。
    fig.savefig(_out, dpi=160, metadata={"Software": None})
    print(f"[图] 已保存 {_out}")

# %% [markdown]
# ## 5. 解码回模态空间
#
# 编码不必无损，但应当能反映原始结构。这里把群体水平编码解回信号值。

# %%
restored = sine_plugin.decode(sine_plugin.encode(signal))
print("原始信号前 5 个值:", np.round(signal[:5], 4))
print("解码回来前 5 个值:", np.round(restored[:5], 4))
print(f"最大误差 {np.abs(restored - signal).max():.4f}（水平编码的量化步长是 {1 / 32:.4f}）")

# %% [markdown]
# ## 下一步
#
# - 想让第三方包提供插件？在你的 `pyproject.toml` 里声明 entry point，用户侧调用
#   `discover_plugins()` 即可，**不需要修改 BioSNN-Plug 任何一行代码**；
# - 想换融合策略？`SpikeBus(strategy=...)` 接受任何满足 `FusionStrategy` 协议的对象。
#   计划书里的 TAAF 时间注意力引导融合将来就从这里接进来。
#
# 插件开发指南见仓库的 `docs/plugin_guide.md`。

# %% [markdown]
# # 论文最小复现：Frémaux et al. (2013) 的 TD-LTP
#
# > Nicolas Frémaux, Henning Sprekeler, Wulfram Gerstner. *Reinforcement Learning Using a
# > Continuous Time Actor-Critic Framework with Spiking Neurons.* PLoS Computational
# > Biology 9(4): e1003024 (2013).
# > [10.1371/journal.pcbi.1003024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024)
#
# 计划书 §12.3 要求「每篇核心文献对应一个 `examples/paper_<name>.py`」。出处由
# **ADR-0007** 结项（计划书 §十一 的 P0 待办）。
#
# 论文对这条规则的命名是逐字的：
#
# > Because it has, roughly, the form of "TD error signal × Hebbian LTP", we call this
# > learning rule **TD-LTP**.
#
# 本脚本演示的是**规则的结构**，不是它在某个任务上的表现：
#
# 1. 三因子形式——前突触活动 × 后突触活动 × **一个全局标量**；
# 2. 权重变化精确等于 `η·δ·e`；
# 3. 成功偏移 / σR 这个验收指标怎么算（计划书 §七：`< 10%σR`）。
#
# **本脚本不演示"有偏信号导致学不会"那条断言。** 论文 §3.2 引 Frémaux et al. (2010)
# 的那条结论（偏移达 ~25%σR 就无法学习）需要设计一个动作分布保持混合的任务才能干净地
# 量出来，本仓库尚未做——见 `research/rstdp/README.md` 的边界一节。写一个量不对的
# 演示比不写更糟。
#
# **运行位置**：必须在本仓库内运行（`research/` 不是分发包，见 ADR-0001/0004）。

# %%
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import torch  # noqa: E402

from research.rstdp.measure_bias import OFFSET_THRESHOLD, BiasTracker  # noqa: E402
from research.rstdp.rstdp import RSTDPActor  # noqa: E402
from research.rstdp.td_ltp import TDLCritic, td_error  # noqa: E402

torch.manual_seed(0)

# %% [markdown]
# ## 1. 两条规则结构相同，第三因子不同
#
# | | 因子 1 | 因子 2 | 因子 3 |
# | :--- | :--- | :--- | :--- |
# | TD-LTP（Critic） | 前突触活动 | 后突触活动 | **TD 误差 δ** |
# | R-STDP（Actor） | 前突触活动 | 被选中的动作 | **成功信号 S = R − b** |
#
# 两者都：非局部量只有**一个标量**。

# %%
critic = TDLCritic(n_features=4, learning_rate=0.1, value_scale=1.0, weight_norm=None)
features = torch.tensor([1.0, 0.5, 0.0, 0.0])
value = critic.value(features)

critic.update(features, value, td_error=1.0)
print(f"Critic 资格痕迹 e = {[round(v, 5) for v in critic.trace.tolist()]}")
print(f"Critic 权重变化 Δw = {[round(v, 5) for v in critic.weights.tolist()]}")

actor = RSTDPActor(n_features=4, n_actions=2, learning_rate=0.1, normalize=False)
before = actor.weights.clone()
actor.update(features, action=0, success_signal=1.0)
change = actor.weights - before
print(f"\nActor Δw 第 0 列（被选中）= {[round(v, 5) for v in change[:, 0].tolist()]}")
print(f"Actor Δw 第 1 列（未选中）= {[round(v, 5) for v in change[:, 1].tolist()]}")

# %% [markdown]
# ## 2. 三因子的可测后果：δ 是全局标量
#
# 「全局」意味着**同一个标量乘在所有突触上**。所以把 δ 翻倍，每个权重的变化量也应当
# 恰好翻倍——这与"每个突触各自有不同的误差信号"是两种结构，可以分辨。


# %%
def trace_and_delta(delta: float):
    probe = TDLCritic(5, learning_rate=1.0, trace_decay=0.5, value_scale=1.0, weight_norm=None)
    f = torch.tensor([1.0, 0.5, -0.5, 0.0, 2.0])
    probe.update(f, probe.value(f), td_error=0.0)  # 先把痕迹建立起来
    before = probe.weights.clone()
    probe.update(f, probe.value(f), td_error=delta)
    return probe.weights - before


once, twice = trace_and_delta(1.0), trace_and_delta(2.0)
torch.testing.assert_close(twice, 2.0 * once, rtol=1e-5, atol=1e-6)
print(f"δ 翻倍后权重变化恰好翻倍：{torch.allclose(twice, 2.0 * once, rtol=1e-5, atol=1e-6)}")

# %% [markdown]
# ## 3. 验收指标怎么算（计划书 §七：成功偏移 < 10%σR）
#
#    偏移  = mean(S)                ← 有符号
#    σR   = std(S)，**总体标准差（ddof = 0）**
#    比值  = |偏移| / σR
#
# 下面用三组**构造好的**信号演示它的判别力——刻意不依赖任何学习过程，因为这里要验的是
# 指标的算法本身。

# %%
cases = {
    "无偏（±1 各半）": [1.0, -1.0] * 50,
    "轻微偏移（±1 各半，加一点点正偏）": [1.0, -1.0] * 50 + [0.2] * 10,
    "强偏移（恒正）": [1.0] * 100,
}
for label, values in cases.items():
    tracker = BiasTracker(tail_window=None)
    tracker.extend(values)
    verdict = "达到" if tracker.overall.passed else "未达到"
    print(
        f"{label:<34} 偏移={tracker.overall.offset:+.4f}  σR={tracker.overall.sigma:.4f}  "
        f"比值={tracker.overall.ratio:.4f}  → {verdict}"
    )

# %% [markdown]
# ## 4. TD 误差的定义
#
# $$\delta = r + \gamma V(s') - V(s)$$

# %%
delta = td_error(reward=1.0, value=torch.tensor(0.5), next_value=torch.tensor(0.7), discount=0.9)
print(f"δ = 1.0 + 0.9×0.7 − 0.5 = {delta:.4f}")
print(
    f"回合终止时 V(s') 取 0 → δ = {td_error(1.0, torch.tensor(0.5), torch.tensor(0.0), discount=0.9):.4f}"
)

# %% [markdown]
# ## 5. 断言

# %%
assert torch.allclose(twice, 2.0 * once, rtol=1e-5, atol=1e-6), "第三因子应当是全局标量"
assert OFFSET_THRESHOLD == 0.10, "阈值取自计划书 §七"
assert BiasTracker(tail_window=None) is not None
assert abs(change[:, 1].sum().item()) < 1e-12, "未被选中的动作不该有权重变化"

print("\n全部断言通过。")

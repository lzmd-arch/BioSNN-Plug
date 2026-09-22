# %% [markdown]
# # 论文最小复现：Bellec et al. (2020) 的 e-prop
#
# > Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj,
# > Robert Legenstein, Wolfgang Maass. *A solution to the learning dilemma for
# > recurrent networks of spiking neurons.* Nature Communications 11, 3625 (2020).
# > [10.1038/s41467-020-17236-y](https://www.nature.com/articles/s41467-020-17236-y)
#
# 计划书 §12.3 要求「每篇核心文献对应一个 `examples/paper_<name>.py`」。本脚本证明那条
# 学习规则在本仓库里真的跑得起来，并且**它为什么是局部的**是可以当场查的。
#
# 刻意**不依赖 MNIST 下载**：用合成的顺序任务，CPU 上几十秒跑完，CI 直接执行。完整的
# sMNIST 验收见 [`research/eprop/train_sequential.py`](../research/eprop/train_sequential.py)。
#
# **运行位置**：必须在本仓库内运行（`research/` 不是分发包，见 ADR-0001/0004）。

# %%
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import torch  # noqa: E402

from research.eprop.eprop import EPropLearner  # noqa: E402
from research.eprop.neurons import ALIFCell, ALIFState  # noqa: E402
from research.eprop.traces import eligibility_traces, eprop_gradient  # noqa: E402

torch.manual_seed(0)

# %% [markdown]
# ## 1. ALIF 动力学
#
# $$v_j^t = \alpha v_j^{t-1} + \sum_i W^{in}_{ji} x_i^t + \sum_i W^{rec}_{ji} z_i^{t-1}
#   - \vartheta\, z_j^{t-1}, \qquad \alpha = e^{-dt/\tau}$$
#
# 论文的式子里 **输入电流上没有 $(1-\alpha)$ 因子**。有些 LIF 实现写成
# $v \leftarrow \alpha v + (1-\alpha) I$，两者只差一个缩放，但会改变有效阈值与不动点。
#
# ALIF 再多一条适应轨迹 $a_j^t = \rho a_j^{t-1} + z_j^{t-1}$，阈值变成
# $\vartheta + \beta a_j^t$。

# %%
cell = ALIFCell(n_in=3, n_rec=4, tau=20.0, tau_adaptation=500.0, threshold=0.62, beta=0.07)
state = ALIFState.zeros(batch_size=1, n_rec=4)
inputs = (torch.rand(20, 1, 3) < 0.3).float()

spikes = []
for t in range(20):
    z, state = cell.step(inputs[t], state)
    spikes.append(z)
spikes = torch.stack(spikes)
print(f"脉冲张量：{tuple(spikes.shape)}，发放率 {spikes.mean():.3f}/步")
print(f"适应变量动了吗：{(state.a > 0).any().item()}（ALIF 的阈值随发放历史上移）")

# %% [markdown]
# ## 2. 资格痕迹（Eq. 25）
#
# $$\psi_j^t = \frac{\kappa}{\vartheta}\max(0, 1-|v^{scaled}_j|), \qquad
#   \varepsilon^v_{ij} \leftarrow \alpha\varepsilon^v_{ij} + z_i^t$$
#
# $$\varepsilon^a_{ij} \leftarrow (\rho - \beta\psi_j)\varepsilon^a_{ij}
#   + \psi_j\varepsilon^v_{ij}, \qquad e_{ij} = \psi_j(\varepsilon^v_{ij} - \beta\varepsilon^a_{ij})$$
#
# $\psi$ 是不应期内的 0/1 掩码乘上伪导数。**不应期判据用的是上一步之后的计数器**——
# 这一步写错不会报错，只会让梯度系统性偏掉。

# %%
v_scaled = torch.stack(
    [cell.v_scaled(ALIFState.zeros(1, 4).v, ALIFState.zeros(1, 4).a)] * 20
)  # 示意用；真实取值来自上面的模拟
trace = eligibility_traces(
    v_scaled,
    torch.cat([torch.zeros_like(spikes[:1]), spikes[:-1]]),
    spikes,
    alpha=cell.alpha,
    rho=cell.rho,
    beta=cell.beta,
    threshold=cell.threshold,
    dampening_factor=cell.dampening_factor,
    n_refractory=cell.n_refractory,
    is_recurrent=True,
)
print(f"资格痕迹：{tuple(trace.shape)} = (T, batch, 前突触, 后突触)")

# %% [markdown]
# ## 3. 关键性质：因子分解什么时候精确、什么时候是近似
#
# e-prop 的定义是 $\Delta W_{ij} = -\eta\sum_t L_j^t e_{ij}^t$。它**不是** BPTT 的梯度。
#
# 差别在于被丢掉的路径：跨神经元的循环路径，以及**复位项** $-\vartheta z^{t-1}$ 带来的
# 自身脉冲介导路径（后者与 $W^{rec}$ 无关，所以把循环权重置零也去不掉）。
#
# 所以下面刻意构造一个"没有可丢掉的路径"的配置——$W^{rec}$ 置零、无不应期、损失只依赖
# 最后一个时间步——此时两者应当**精确相等**。这是对公式与时间对齐的硬检验。

# %%
torch.manual_seed(1)
steps, batch, n_in, n_rec = 6, 1, 3, 4
probe = ALIFCell(n_in, n_rec, beta=0.0, n_refractory=0)
with torch.no_grad():
    probe.w_rec.zero_()
probe_inputs = (torch.rand(steps, batch, n_in) < 0.5).float()

state = ALIFState.zeros(batch, n_rec)
probe_spikes, probe_scaled = [], []
for t in range(steps):
    z, state = probe.step(probe_inputs[t], state)
    probe_spikes.append(z)
    probe_scaled.append(probe.v_scaled(state.v, state.a))
probe_spikes = torch.stack(probe_spikes)
probe_scaled = torch.stack(probe_scaled)

loss = probe_spikes[-1].pow(2).sum()  # 只看最后一步 → 更早的复位路径影响不到它
learning_signal = torch.autograd.grad(loss, probe_spikes, retain_graph=True)[0]
z_prev = torch.cat([torch.zeros_like(probe_spikes[:1]), probe_spikes[:-1]]).detach()

grad_eprop = eprop_gradient(
    probe_scaled.detach(),
    z_prev,
    probe_spikes.detach(),
    learning_signal,
    alpha=probe.alpha,
    rho=probe.rho,
    beta=probe.beta,
    threshold=probe.threshold,
    dampening_factor=probe.dampening_factor,
    n_refractory=0,
    is_recurrent=True,
)
grad_bptt = torch.autograd.grad(loss, probe.w_rec)[0]
relative = (grad_eprop - grad_bptt).abs().max().item() / max(grad_bptt.abs().max().item(), 1e-12)
print(f"无循环、无不应期、单步损失时的相对差：{relative:.3e}（应当是 0 或机器精度）")

# %% [markdown]
# ## 4. 学习路径上没有 autograd
#
# 这是计划书 §9「无全局反向传播」的落点。学习信号有**闭式**：
# $\partial E/\partial z^{filt} = W^{out\top}(\mathrm{softmax}(\hat y) - y^*)$，
# 一行矩阵乘法。所以整个学习路径没有一次 `backward()`。

# %%
model = EPropLearner(
    n_in=6,
    n_rec=32,
    n_out=2,
    learning_rate_in=5e-3,
    learning_rate_rec=5e-3,
    learning_rate_out=5e-2,
    generator=torch.Generator().manual_seed(2),
)

# 顺序任务：判别信息只在序列**后半段**出现，考验时序记忆
task_steps, task_batch = 8, 16
labels = torch.arange(task_batch) % 2
task_inputs = torch.zeros(task_steps, task_batch, 6)
task_inputs[task_steps // 2 :, :, 0] = (
    (labels == 0).float().unsqueeze(0).expand(task_steps - task_steps // 2, task_batch)
)
task_inputs[task_steps // 2 :, :, 1] = (
    (labels == 1).float().unsqueeze(0).expand(task_steps - task_steps // 2, task_batch)
)

first = model.loss_value(model.simulate(task_inputs), labels)
for _ in range(80):
    last = model.update(task_inputs, labels)
accuracy = float((model.predict(task_inputs) == labels).float().mean())
print(f"交叉熵：{first:.4f} → {last:.4f}")
print(f"准确率：{accuracy:.3f}（随机基线 0.50）")
print(f"更新后 w_rec.grad 是 None 吗：{model.cell.w_rec.grad is None}（无 autograd 参与）")

# %% [markdown]
# ## 5. §9 的活跃神经元比例
#
# 这是**原义**指标（脉冲网络）：整个评估窗内至少发放过一次的神经元占比。§3.1 规定低于
# 60% 触发阈值调整，§七 第一阶段要求 > 60%。
#
# **注意这个示例的配置极小**（32 个神经元、80 步更新），活跃比例会明显低于验收配置——
# 验收配置（256 个神经元、30 个 epoch）实测是 1.0000。这里只打印，不做断言。

# %%
active = model.active_neuron_fraction(task_inputs)
sparsity = model.spike_sparsity(task_inputs)
print(f"活跃神经元比例：{active:.4f}")
print(f"脉冲稀疏度：{sparsity:.4f}")

# %% [markdown]
# ## 6. 断言

# %%
assert relative < 1e-5, f"因子分解应当精确，实测相对差 {relative:.3e}"
assert model.cell.w_rec.grad is None, "学习路径上不该有 autograd"
assert accuracy > 0.5 + 0.2, f"e-prop 没有学到东西：准确率仅 {accuracy:.3f}"

print("\n全部断言通过。")

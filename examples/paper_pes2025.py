# %% [markdown]
# # 论文最小复现：Pes et al. (2025) 的 Trace Propagation
#
# > Luca Pes, Bojian Yin, Sander Stuijk, Federico Corradi. *Traces Propagation:
# > Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks.*
# > arXiv:2509.13053；期刊版 *Neuromorphic Computing and Engineering* **6**(1):014002 (2026).
# > [10.1088/2634-4386/ae2ef9](https://doi.org/10.1088/2634-4386/ae2ef9)
#
# 计划书 §12.3 要求「每篇核心文献对应一个 `examples/paper_<name>.py`」，用来拦「论文说能跑
# 但仓库跑不通」。这是 **Pes 2025 的最小复现**：把 TP 的规则本身在玩具任务上跑起来。
#
# **先说清楚它不是什么。** TP **不是**本仓库 W2 用的学习规则。计划书 §3.1 曾把它写成
# 「e-prop 的存储优化：O(N²) → O(N)」，逐条查原文之后**不采纳**——它是**另一条规则**
# （痕迹按神经元而非按突触存），论文里它的**全部**实验都是 LIF，覆盖不到 W2 的 ALIF 配置。
# 裁决与三条证据见 [`docs/adr/ADR-0010`](../docs/adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.md)。
# 所以本脚本**只验 TP 这条规则自己**，不与 `research/eprop/` 的实现比精度、也不进那条线。
#
# **这个脚本的第二个用途**：论文的公式与它自己的官方实现有几处对不上，照论文写会得到另一个
# 算法。每一处下面都标了【论文 vs 代码】，这是复现里最值钱的部分。
#
# 刻意**不依赖任何下载**：合成任务，CPU 上十几秒跑完，CI 直接执行。

# %%
import math

import torch

#: 本脚本的随机性来源。**不用研究代码的种子簿**（`research/common/seeding.py`）：这是
#: examples/ 下的独立复现脚本，要能被 CI 直接执行、不与研究线耦合，所以用一个显式构造的
#: generator，种子写在脚本里。
GENERATOR = torch.Generator().manual_seed(20250923)

# %% [markdown]
# ## 1. LIF 动力学，以及论文 Eq. (1) 的一处 off-by-one
#
# 论文 Eq. (1)(2)（逐字）：
#
# $$v_l^t[j] = \alpha_l v_l^{t-1}[j] + s_{l-1}^t[i]\,W_l[i,j] - s_l^{t-1}[j]\,v_{th},
#   \qquad s_l^t[j] = \Theta\!\left(v_l^t[j] - v_{th}\right)$$
#
# **但官方实现用的是即时软复位**（`models/neuron_layers.py::LIFLayer.forward`）：
#
# ```python
# pre = leak_m * v + input_current      # 先积分
# s   = activation(pre)                 # 再发放：判据用的是**复位前**的 pre
# v   = pre - s * vth                   # 再复位：用**当前**脉冲，不是 s^{t-1}
# ```
#
# 也就是说 Eq. (1) 里那个 $-s_l^{t-1}v_{th}$ 是 off-by-one，**照代码写**。这一处差异会改变
# 整个动力学（复位发生在发放之前还是之后），所以下面两处都标出来。
#
# 伪导数：论文只写了一句散文——
#
# > As a surrogate function $\theta'$, we employ the ArcTan function defined in
# > [Fang et al. 2021] with a scale factor of 1.
#
# ——**没有给公式**。官方仓库的 `surrogate_type="1"` 是
# $\Theta'(z) = \dfrac{1}{1 + (\pi z)^2}$，$z = v - v_{th}$。（注意 `main.py` 的 argparse
# 默认是 `surrogate_scale=10.0`，而 N-MNIST / SHD 的实验脚本都显式覆盖成 1——别用默认值。）


# %%
def pseudo_derivative(v_pre: torch.Tensor, v_th: float) -> torch.Tensor:
    r"""ArcTan 代理梯度 $\Theta'(v - v_{th}) = 1 / (1 + (\pi (v - v_{th}))^2)$。"""
    z = v_pre - v_th
    return 1.0 / (1.0 + (math.pi * z) ** 2)


class SurrogateSpike(torch.autograd.Function):
    r"""$\Theta(v - v_{th})$，反传时换成 ArcTan 代理梯度。

    **这一步不是可选的。** 阶跃函数本身没有梯度，而 Eq. (18) 里那个
    $\Theta'(v_l^t - V_{\mathrm{th}})$ 就是它——不把代理梯度接进计算图，下面的损失会直接报
    `does not require grad`。所以「让 autograd 展开 Eq. (18)」这句话的前提是：脉冲张量
    带着代理梯度。
    """

    @staticmethod
    def forward(ctx, v_pre: torch.Tensor, v_th: float) -> torch.Tensor:
        ctx.save_for_backward(v_pre)
        ctx.v_th = v_th  # type: ignore[attr-defined]
        return (v_pre > v_th).to(v_pre.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_pre,) = ctx.saved_tensors
        return grad_output * pseudo_derivative(v_pre, ctx.v_th), None  # type: ignore[attr-defined]


# %% [markdown]
# ## 2. 两条逐神经元痕迹（Eq. 11/12）
#
# $$\epsilon_l^t[b,j] = \beta_l\,\epsilon_l^{t-1}[b,j] + s_l^t[b,j] \tag{11}$$
# $$\tilde{\epsilon}_l^t[b,j] = \beta_l\,\tilde{\epsilon}_l^{t-1}[b,j] + \tilde{s}_l^t[b,j] \tag{12}$$
#
# 形状是 **`(B, H_l)`：batch × 本层神经元**——逐神经元，不是逐突触。这一条就是论文 Table 3
# 里 TP 那行 `Space = LH` 的全部来源（对照 e-prop 行的 `LH²`）。论文原话：
#
# > On the other hand, algorithms such as ETLP, S-TLLR, DECOLLE and TESS store traces per
# > neuron, leading to a reduced space complexity of $\mathcal{O}(LH)$. Similarly, as defined in
# > Equations 11 and 12, TP requires two traces that are stored per neuron, one for the input
# > signal and another for the target signal, leading to a total memory cost of $2BLH$ and a
# > complexity of $\mathcal{O}(LH)$.
#
# 第 7 节会把这两种存储量级**按本脚本的配置实际数一遍**并断言。


# %%
class TracePropLayer:
    """一层 LIF，带 TP 的两条逐神经元痕迹。

    **两条通路各有状态，权重共享与否取决于层号**：

    * 输入通路用 ``weight``（``n_in → n_out``）；
    * 目标通路在 **l = 1** 用 ``target_weight``，也就是论文那张 ``S``（``C → H_1``，
      官方实现里默认冻结）；在 **l > 1** 用**同一份** ``weight``——论文原话是
      "the same matrices $W_l$ are used to propagate the target signal"。
      所以这一层有两种构造方式，由 ``target_n_in`` 决定。

    Attributes:
        trace: ``(B, n_out)``，输入痕迹 ``ε_l``（Eq. 11）。
        trace_target: ``(B, n_out)``，目标痕迹 ``ε̃_l``（Eq. 12）。
        last_spike: 最近一步的脉冲（诊断用）。
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        target_n_in: int | None = None,
        alpha: float,
        beta: float,
        v_th: float,
        generator: torch.Generator,
    ) -> None:
        self.n_in, self.n_out = n_in, n_out
        self.alpha, self.beta, self.v_th = alpha, beta, v_th
        # 论文没写权重初始化；官方仓库用 PyTorch 的 nn.Linear 默认 U(-1/√n, 1/√n)。
        bound = 1 / math.sqrt(n_in)
        self.weight = torch.nn.Parameter(
            torch.empty(n_in, n_out).uniform_(-bound, bound, generator=generator)
        )
        if target_n_in is None:
            # l > 1：目标通路复用输入通路的权重（同一个 Parameter 对象，不是副本）
            self.target_weight = self.weight
        else:
            # l = 1：目标通路是那张 S。论文 Algorithm 1 把这一情形写成 `W_l = S`；
            # 官方实现默认 `train_s=False`（冻结），本脚本照做。
            bound = 1 / math.sqrt(target_n_in)
            self.target_weight = torch.empty(target_n_in, n_out).uniform_(
                -bound, bound, generator=generator
            )
        self.reset(batch_size=1)

    def reset(self, batch_size: int) -> None:
        """每个 batch 开始时清状态（官方实现的 ``model.reset_potential()``）。"""
        zeros = lambda: torch.zeros(batch_size, self.n_out)  # noqa: E731
        self.v, self.v_target = zeros(), zeros()
        self.trace, self.trace_target = zeros(), zeros()
        self.last_spike = zeros()

    def step(self, spike_in: torch.Tensor, spike_target_in: torch.Tensor):
        """一个时间步。两个输入必须是**已 detach** 的上游脉冲（或原始输入与 one-hot 标签）。

        Returns:
            ``(spike, trace, target_spike, target_trace)``；脉冲 detach 过（它要进下一层），
            两条痕迹带计算图（本层的局部损失要用）。
        """
        # **跨层 detach 是 TP 成立的前提。** 论文只隐含（Eq. 18 只对 θ_l 求导、不含上游），
        # 官方实现是显式的 `spike.detach()`；漏掉它，autograd 会顺着跨层递推一路反传回去，
        # TP 就退化成 BPTT 了。第 8 节有断言盯着这件事。
        spike_in = spike_in.detach()
        spike_target_in = spike_target_in.detach()
        v_prev, v_target_prev = self.v.detach(), self.v_target.detach()
        trace_prev, trace_target_prev = self.trace.detach(), self.trace_target.detach()

        # 输入通路：先积分 → 发放（判据用**复位前**的 pre，代理梯度见第 1 节）→ 软复位
        v_pre = self.alpha * v_prev + spike_in @ self.weight
        spike = SurrogateSpike.apply(v_pre, self.v_th)
        self.v = v_pre.detach() - self.v_th * spike.detach()

        # 目标通路：另一条并行的状态
        v_target_pre = self.alpha * v_target_prev + spike_target_in @ self.target_weight
        spike_target = SurrogateSpike.apply(v_target_pre, self.v_th)
        self.v_target = v_target_pre.detach() - self.v_th * spike_target.detach()

        # 痕迹（Eq. 11/12）：逐神经元，形状 (B, n_out)
        self.trace = self.beta * trace_prev + spike
        self.trace_target = self.beta * trace_target_prev + spike_target
        self.last_spike = spike.detach()
        return spike.detach(), self.trace, spike_target.detach(), self.trace_target


# %% [markdown]
# ## 3. 目标通路：一张 `S` 矩阵，其余层共用 `W_l`
#
# 论文 Fig. 2 的图注（逐字）：
#
# > In the purple path, the one-hot encoded target vector $c \in \mathcal{R}^C$ is projected to
# > the first layer via $S \in \mathcal{R}^{C \times H_1}$, to match the dimensionality of the
# > first input trace (i.e. $\epsilon_1^t$) … For $l>1$, the same matrices $W_l$ are used to
# > propagate the target signal and generate target traces $\tilde{\epsilon}_l^t$ at each layer.
#
# 所以目标通路是**一条并行的前向通路**，额外的参数只有一张 `S`——这也是 Table 3 里辅助矩阵
# 从 `LOH`（每层一张 `B_l`）降到 `OH` 的原因。
#
# **喂进目标通路的是 one-hot 本身，不是它的痕迹**（Algorithm 1 第 16 行：`s̃_0^t = c`）。
# 那个痕迹 `ε̃_0` 另有用途：它是 Eq. (15) 里 l=1 那一层的 `ε̃_{l-1}`。
#
# 【论文 vs 代码】官方实现里 `S` 是一个**独立的 LIF 层**（`self.target_propagator`，带自己的
# 膜电位与阈值），默认 `train_s=False`（冻结）。本脚本照做：`S` 冻结，但仍参与前向。
#
# ## 4. 对比损失（Eq. 13/14/15）
#
# $$E_l^t = -y_l^t[b,b'] \log\!\left(\mathrm{Softmax}\!\left(z_l^t[b,b']\right)\right) \tag{13}$$
# $$z_l^t[b,b'] = \epsilon_l^t[b,j]\,\tilde{\epsilon}_l^t[j,b'] \tag{14}$$
# $$y_l^t[b,b'] = \mathrm{Softmax}\!\left(f\!\left(\tilde{\epsilon}_{l-1}^t[b,j],
#   \tilde{\epsilon}_{l-1}^t[j,b']\right)\right) \tag{15}$$
#
# 三处最容易写错的地方：
#
# 1. **`z` 用本层的两条痕迹，`y` 用的是上一层的目标痕迹**（Eq. 15 的下标是 $l-1$）；
# 2. 两个量都是 **`B × B`**（batch 内的两两相似度），所以**论文要求 `B ≥ 2`**，而且
#    **逐样本在线更新做不到**——这是论文自认的第一条局限；
# 3. 论文**没有**给 `f` 定死（"e.g. dot product or negative euclidean distance"），也
#    **没有**温度系数或 L2 归一化。【论文 vs 代码】官方取**负欧氏距离**：
#    $f = -\lVert \tilde{\epsilon}_{l-1}[b] - \tilde{\epsilon}_{l-1}[b'] \rVert_2$。
#
# 顺带记一笔论文内部的不一致：复杂度小节把调制信号写成 $(y_l^t - z_l^t)$，而 Eq. (18) 与
# Algorithm 1 写的是 $(\mathrm{Softmax}(z_l^t) - y_l^t)$——**符号相反**。以 Eq. (18) 为准
# （带 `log_softmax` 的交叉熵，梯度确实是 $\mathrm{Softmax}(z) - y$）。


# %%
def contrastive_loss(
    trace: torch.Tensor, trace_target: torch.Tensor, trace_target_below: torch.Tensor
) -> torch.Tensor:
    """一层、一个时间步的局部损失（Eq. 13-15）。

    Args:
        trace: ``ε_l``，``(B, H_l)``。
        trace_target: ``ε̃_l``，``(B, H_l)``。
        trace_target_below: ``ε̃_{l-1}``，``(B, H_{l-1})``——**上一层**的目标痕迹。
    """
    logits = trace @ trace_target.t()  # Eq. (14)：B × B
    # Eq. (15)：f = 负欧氏距离（官方实现的选择），对 b' 维做 row-wise softmax
    distance = torch.cdist(trace_target_below.detach(), trace_target_below.detach())
    targets = torch.softmax(-distance, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()  # Eq. (13)


# %% [markdown]
# ## 5. 更新式（Eq. 16-18）与输出层
#
# Eq. (18)（论文三行 align 的最后一行，逐字）：
#
# $$\frac{\partial E_l^t}{\partial \theta_l[i,j]} =
#   \left(\mathrm{Softmax}(z_l^t) - y_l^t\right)[b,b']\Big[\,
#   \tilde{\epsilon}_l^t[b',j]\,\Theta'(v_l^t[b,j] - V_{\mathrm{th}})\,s_{l-1}^t[b,i]
#   + \epsilon_l^t[b,j]\,\Theta'(\tilde{v}_l^t[b',j] - V_{\mathrm{th}})\,\tilde{s}_{l-1}^t[b',i]\,\Big]$$
#
# 两项分别对应**输入通路**与**目标通路**——两条通路共用 `W_l`，所以它俩必然同时出现。
# 伪导数在**当前时刻**求值，自变量是**复位前**的膜电位（见第 1 节）。
#
# **本脚本不手写这个式子，而是让 autograd 去展开它**：只要 (a) 跨层脉冲 detach、
# (b) 上一时刻的状态 detach、(c) 脉冲带代理梯度，autograd 给出的就是 Eq. (18)。
# 这样更不容易错，而且「detach 是不是真的切断了」可以当场断言（第 8 节）。
#
# 输出层论文里是「积分器」（"a simple integrator as an output layer, where the predicted class
# corresponds to the neuron with the highest integration value at the end of the sequence"），
# 但 **Algorithm 1 完全没有它的更新式**。【论文 vs 代码】官方实现用输出层的交叉熵局部梯度：
#
# $$\Delta W_{out} = \frac{1}{B}\,\epsilon_L^\top\left(\mathrm{Softmax}(m) - c\right)$$
#
# 少了这条复现不出来，所以本脚本照代码补上。


# %%
class TracePropNetwork:
    """TP 网络：``输入 → LIF(痕迹) ×L → 线性积分器``。

    用**两层**隐藏层，是为了让论文那句「l>1 时目标通路复用同一份 W_l」真的被执行到——
    单隐层的话这句话在本脚本里是个空条款。
    """

    def __init__(
        self,
        n_in: int,
        hidden_sizes: tuple[int, ...],
        n_classes: int,
        *,
        alpha: float = 0.9,
        beta: float = 0.9,
        v_th: float = 1.0,
        generator: torch.Generator,
    ) -> None:
        self.n_classes = n_classes
        self.beta = beta
        self.layers: list[TracePropLayer] = []
        sizes = (n_in, *hidden_sizes)
        for index in range(len(hidden_sizes)):
            self.layers.append(
                TracePropLayer(
                    sizes[index],
                    hidden_sizes[index],
                    # 只有第一层的目标通路换权重（那张 S）；其余共用输入通路的 W_l
                    target_n_in=n_classes if index == 0 else None,
                    alpha=alpha,
                    beta=beta,
                    v_th=v_th,
                    generator=generator,
                )
            )
        bound = 1 / math.sqrt(hidden_sizes[-1])
        self.readout = torch.nn.Parameter(
            torch.empty(hidden_sizes[-1], n_classes).uniform_(-bound, bound, generator=generator)
        )

    def parameters(self) -> list[torch.Tensor]:
        return [layer.weight for layer in self.layers] + [self.readout]

    def reset(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset(batch_size)
        # 第 0 层的两条痕迹（ε_0 与 ε̃_0）——它们只进损失，不进任何一层
        self.trace_in = torch.zeros(batch_size, self.layers[0].n_in)
        self.trace_target_in = torch.zeros(batch_size, self.n_classes)
        self.integration = torch.zeros(batch_size, self.n_classes)

    def forward_step(self, inputs: torch.Tensor, one_hot: torch.Tensor):
        """一个时间步：两条通路 → 逐层局部损失 → 输出层积分。

        Returns:
            ``(losses, output_grad, integration)``——``losses`` 是逐层的对比损失（Eq. 13），
            ``output_grad`` 是输出层的局部梯度（论文 Algorithm 1 里没有这条，见上）。
        """
        # 第 0 层的两条痕迹：ε_0 = β·ε_0 + x，ε̃_0 = β·ε̃_0 + c（Algorithm 1 第 2-3 行）
        self.trace_in = self.beta * self.trace_in.detach() + inputs
        self.trace_target_in = self.beta * self.trace_target_in.detach() + one_hot

        spike, spike_target = inputs, one_hot
        trace_target_below = self.trace_target_in  # Eq. (15) 里的 ε̃_{l-1}；l=1 时就是 ε̃_0
        losses: list[torch.Tensor] = []
        last_trace = self.trace_in
        for layer in self.layers:
            spike, trace, spike_target, trace_target = layer.step(spike, spike_target)
            losses.append(contrastive_loss(trace, trace_target, trace_target_below))
            trace_target_below = trace_target
            last_trace = trace

        # 输出积分器：m ← m + ε_L W_out（预测类别 = 序列末积分值最大的那个神经元）
        self.integration = self.integration.detach() + last_trace @ self.readout
        error = torch.softmax(self.integration, dim=-1) - one_hot
        # 【论文 vs 代码】官方实现显式写出输出层的梯度（不走 autograd）：
        # ΔW_out = (1/B)·ε_Lᵀ·(Softmax(m) − c)
        output_grad = last_trace.detach().t() @ error / inputs.shape[0]
        return losses, output_grad, self.integration


# %% [markdown]
# ## 6. 玩具任务
#
# 每个类别占一块**专属输入通道**，块内的时空图样逐类不同，再加 5% 的漏脉冲噪声。判别信息
# 落在通道上而不是时序上——这与论文用的 N-MNIST / SHD 是同一种情形（那些数据集的信息也在
# 通道上；论文的积分器读出本身就只累加痕迹，对时序先后不敏感）。
#
# **任务本身是本项目自己的设计**，不是论文的任何实验设置。
#
# 先记一个数字：这个任务上「输入脉冲累加 + 线性分类器」能到 **1.000**——它是可分性的上限，
# 也是后面「TP 学到了东西」的参照。而把隐藏层**随机冻结**、只训读出，只能到 **0.31**
# （下面的第 8 节会断言完整 TP 明显高于它）——所以这里的 TP 确实在塑造特征，不是白捡。


# %%
def make_task(n_classes: int, steps: int, n_in: int, n_samples: int, generator: torch.Generator):
    per_class = n_in // n_classes
    templates = torch.zeros(n_classes, steps, n_in)
    for label in range(n_classes):
        block = slice(label * per_class, (label + 1) * per_class)
        templates[label, :, block] = (
            torch.rand(steps, per_class, generator=generator) < 0.5
        ).float()
    labels = torch.randint(0, n_classes, (n_samples,), generator=generator)
    inputs = templates[labels].clone()
    dropout = (torch.rand(inputs.shape, generator=generator) < 0.05).float()
    return inputs * (1 - dropout), labels


# %%
STEPS, N_IN, N_CLASSES, N_HIDDEN, BATCH = 12, 32, 4, 64, 32
EPOCHS, BATCHES_PER_EPOCH, LEARNING_RATE = 20, 16, 1e-2

train_inputs, train_labels = make_task(
    N_CLASSES, STEPS, N_IN, BATCH * BATCHES_PER_EPOCH * EPOCHS, GENERATOR
)
test_inputs, test_labels = make_task(N_CLASSES, STEPS, N_IN, 512, GENERATOR)

network = TracePropNetwork(N_IN, (N_HIDDEN, N_HIDDEN), N_CLASSES, generator=GENERATOR)
optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

# %% [markdown]
# **训练一步的顺序**（官方 `utils.py::train` 的 `--T 1` 配置，即"每个时间步都更新"）：
#
# ```text
# model.reset_potential()      # 膜电位 / 痕迹 / 上游脉冲归零
# for t in range(T):
#     forward(...)             # 两条通路
#     model.update()           # 逐层 autograd.grad(局部损失, 本层 W)
#     optimizer.step()         # ← 序列内逐步更新（在线）
#     model.detach_membrane_states()
# ```
#
# `--T -1` 才是"把整段序列的更新加起来"。论文的实验脚本用的是 `--T 1`。

# %%
history = []
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for batch_index in range(BATCHES_PER_EPOCH):
        start = ((epoch - 1) * BATCHES_PER_EPOCH + batch_index) * BATCH
        inputs = train_inputs[start : start + BATCH]
        one_hot = torch.nn.functional.one_hot(
            train_labels[start : start + BATCH], N_CLASSES
        ).float()

        network.reset(inputs.shape[0])
        for t in range(STEPS):
            optimizer.zero_grad()
            losses, output_grad, _ = network.forward_step(inputs[:, t], one_hot)
            sum(losses).backward()  # Eq. (18) 由 autograd 展开；图在层间与时间上都是断开的
            network.readout.grad = output_grad
            optimizer.step()  # `--T 1`：每个时间步都更新
            epoch_loss += float(sum(losses).detach())
    history.append(epoch_loss / (BATCHES_PER_EPOCH * STEPS))
    print(f"epoch {epoch:2d}  平均对比损失 {history[-1]:.4f}")

# %%
with torch.no_grad():
    network.reset(len(test_labels))
    for layer in network.layers:
        layer.spike_rate = 0.0
    for t in range(STEPS):
        network.forward_step(test_inputs[:, t], torch.zeros(len(test_labels), N_CLASSES))
        for layer in network.layers:
            layer.spike_rate += float(layer.last_spike.mean()) / STEPS  # type: ignore[attr-defined]
    prediction = network.integration.argmax(dim=-1)
    accuracy = float((prediction == test_labels).float().mean())

print(f"\n平均对比损失：{history[0]:.4f} → {history[-1]:.4f}")
print(f"测试准确率：{accuracy:.3f}（{N_CLASSES} 类随机基线 {1 / N_CLASSES:.2f}）")
for index, layer in enumerate(network.layers):
    print(f"  隐藏层 {index + 1}：发放率 {layer.spike_rate:.3f}")  # type: ignore[attr-defined]

# %% [markdown]
# **别把对比损失的绝对值读成准确率。** 它只从 5.03 降到 4.81，而分类准确率是 1.000——
# 这两件事在这里不是一回事：对比损失是 `B × B` 尺度的量（B=32 时它的"均匀预测"水平就是
# `ln 32 ≈ 3.47`），它管的是「同一类样本的痕迹互相靠拢」，而**分类**是读出层自己那条交叉熵
# 梯度（论文没写的那一条）在做。所以本脚本对"学会了"的判据是**准确率**，不是损失值。

# %% [markdown]
# ## 7. 存储量级：论文那条主张按本脚本的配置实际数一遍
#
# Table 3 里 TP 行的 Space Complexity 是 `LH`（对照 e-prop 行的 `LH²`）。把两种存法在本脚本
# 的配置下**实际数出来**，比引用表格更有说服力：TP 的两条痕迹是 `2·B·ΣH`，而按突触存
# （e-prop 的 `ε_ij`）是 `B·ΣH²`。

# %%
n_layers = len(network.layers)
per_neuron = 2 * BATCH * N_HIDDEN * n_layers
per_synapse = BATCH * N_HIDDEN * N_HIDDEN * n_layers
print(f"TP 的痕迹状态（逐神经元，2·B·ΣH）  ：{per_neuron:>9,} 个元素")
print(f"        同样规模按突触存（逐突触，B·ΣH²）：{per_synapse:>9,} 个元素")
print(f"        比值 {per_synapse / per_neuron:.0f}×（= H/2，与 Table 3 的 LH vs LH² 同源）")

# %% [markdown]
# ### 对照：把隐藏层冻住，只训读出
#
# 上面那个 1.000 里，读出层有自己的交叉熵梯度（论文没写的那一条），所以必须回答一个问题：
# **是不是读出单独就把任务做完了？** 把隐藏层换成同样初始化的随机权重、冻结不动，只训
# 读出层，跑同样的步数——这就是答案。

# %%
control = TracePropNetwork(
    N_IN, (N_HIDDEN, N_HIDDEN), N_CLASSES, generator=torch.Generator().manual_seed(20250923)
)
control_optimizer = torch.optim.Adam([control.readout], lr=LEARNING_RATE)
for epoch in range(1, EPOCHS + 1):
    for batch_index in range(BATCHES_PER_EPOCH):
        start = ((epoch - 1) * BATCHES_PER_EPOCH + batch_index) * BATCH
        inputs = train_inputs[start : start + BATCH]
        one_hot = torch.nn.functional.one_hot(
            train_labels[start : start + BATCH], N_CLASSES
        ).float()
        control.reset(inputs.shape[0])
        with torch.no_grad():
            for t in range(STEPS):
                _, output_grad, _ = control.forward_step(inputs[:, t], one_hot)
                control_optimizer.zero_grad()
                control.readout.grad = output_grad
                control_optimizer.step()

with torch.no_grad():
    control.reset(len(test_labels))
    for t in range(STEPS):
        control.forward_step(test_inputs[:, t], torch.zeros(len(test_labels), N_CLASSES))
    control_accuracy = float(control.integration.argmax(dim=-1).eq(test_labels).float().mean())

print(f"对照（隐藏层随机冻结，只训读出）：{control_accuracy:.3f}")
print(f"完整 TP（隐藏层由对比损失塑造）：{accuracy:.3f}")

# %% [markdown]
# ## 8. 断言

# %%
# 局部性：**上游脉冲必须被 detach**。漏掉这一步，梯度会跨层回传，TP 就退化成 BPTT。
# 这条断言直接查 `step()` 的实现：给一个 requires_grad 的输入，反传后它不该拿到梯度。
probe = TracePropLayer(4, 4, alpha=0.9, beta=0.9, v_th=1.0, generator=GENERATOR)
spike_in = torch.ones(2, 4, requires_grad=True)
probe.step(spike_in, torch.zeros(2, 4))[1].sum().backward()
assert spike_in.grad is None, "上游脉冲必须被 detach：否则梯度跨层回传，TP 就退化成 BPTT"

# 学习：损失要降、准确率要显著高于随机
assert history[-1] < history[0], f"对比损失没有下降：{history[0]:.4f} → {history[-1]:.4f}"
assert accuracy > 2 / N_CLASSES, f"TP 没有学到东西：准确率仅 {accuracy:.3f}"

# **TP 的规则确实在塑造特征**：完整训练的准确率要明显高于「隐藏层冻结、只训读出」那条对照。
# 这条是上面那个 1.000 的防伪标记——没有它，读者无法区分「TP 学会了」与「读出学会了」。
assert accuracy > control_accuracy + 0.3, (
    f"完整 TP {accuracy:.3f} 与只训读出的对照 {control_accuracy:.3f} 差得不够多，"
    "说不清是 TP 在塑造特征还是读出单独在做"
)

# 存储：逐神经元的痕迹必须确实比逐突触小（这是 TP 的卖点，不是修辞）
assert per_neuron < per_synapse

print("\n全部断言通过。")

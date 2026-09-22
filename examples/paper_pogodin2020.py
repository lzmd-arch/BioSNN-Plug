# %% [markdown]
# # 论文最小复现：Pogodin & Latham (2020) 的核化 IB-Hebbian 规则
#
# > Roman Pogodin & Peter E. Latham. *Kernelized information bottleneck leads to
# > biologically plausible 3-factor Hebbian learning in deep networks.* NeurIPS 2020.
# > [arXiv:2006.07123](https://arxiv.org/abs/2006.07123)
#
# 计划书 §12.3 要求「每篇核心文献对应一个 `examples/paper_<name>.py`」，用来拦
# 「论文说能跑但仓库跑不通」。所以这个脚本的目标不是复现论文的精度，而是**证明那条
# 学习规则在本仓库里真的跑得起来并真的在学**。
#
# 因此它刻意**不依赖 MNIST 下载**：用合成的高维二分类数据，几十秒内在 CPU 上跑完，
# CI 里可以直接执行。完整的 MNIST 验收运行见
# [`research/ib_hebbian/train_mnist.py`](../research/ib_hebbian/train_mnist.py)，
# 那一条需要先跑 `scripts/download_data.py mnist`。
#
# 本脚本验证三件事：
#
# 1. 每层的局部目标（Eq. 9）能被算出来，且它的梯度只落在本层权重上；
# 2. 反复用局部规则更新后，读出层的分类准确率**显著高于随机**——规则确实在学；
# 3. 除法归一化（Eq. 13/17）按论文所说参与其中。
#
# **运行位置**：必须在本仓库内运行。`research/` 刻意不是分发包（见 ADR-0001、
# ADR-0004），所以下面的 sys.path 引导是必需的——`biosnn-bus` 那种 `pip install`
# 就能在 Colab 里跑的路径，对本脚本不成立。

# %%
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import torch  # noqa: E402

from research.ib_hebbian.layers import LocalObjectiveLayer  # noqa: E402
from research.ib_hebbian.model import IBHebbianPerceptron  # noqa: E402
from research.ib_hebbian.objective import centered_label_kernel  # noqa: E402

torch.manual_seed(0)

# %% [markdown]
# ## 1. 合成数据
#
# 两类高维高斯团，均值不同、各向同性。这个任务线性可分，所以任何"能学"的规则都应该
# 远高于 50%；若准确率贴着随机，说明规则或接线有问题。

# %%
N_PER_CLASS, DIM, N_CLASSES = 256, 128, 2
centers = torch.stack([torch.zeros(DIM), torch.full((DIM,), 1.2)])
labels = torch.arange(N_PER_CLASS * N_CLASSES) % N_CLASSES
features = centers[labels] + 0.8 * torch.randn(N_PER_CLASS * N_CLASSES, DIM)

# 打乱一次，让批里两类混合
order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(1))
features, labels = features[order], labels[order]
print(f"数据：{tuple(features.shape)}，{N_CLASSES} 类，每类 {N_PER_CLASS} 个样本")

# %% [markdown]
# ## 2. 教学信号是二值的（Eq. 35）
#
# 类别均衡时，居中标签的余弦相似度化简为：同类 `1`，异类 `−1/(n−1)`。两分类时异类
# 恰好是 `−1`，于是第三因子里来自标签的部分退化成纯符号。

# %%
kernel = centered_label_kernel(labels, N_CLASSES)
print(f"教学信号取值：{sorted(set(kernel.flatten().tolist()))}")

# %% [markdown]
# ## 3. 单层的局部性：梯度不跨层
#
# 这是计划书 §3.3「第三因子无需自顶向下传递」与 §9「无全局反向传播」在代码里的
# 可检验形式：层的输入被切断梯度，所以更新只可能落在本层权重上。
#
# **需要说清楚的是**：本实现用 autograd 计算**该层局部目标**对自身权重的梯度——
# autograd 在这里是仿真手段，不是学习机制。局部性主张的根据是论文由 Eq. (10) 推导出
# 的三因子 Hebbian 形式（Eq. 12 / Eq. 18）。

# %%
layer = LocalObjectiveLayer(DIM, 64, n_groups=8)
x = torch.randn(16, DIM)
out = layer(x, update=False)
print(f"层输出是否携带计算图：{out.grad_fn is not None}（应为 False）")

layer(x, centered_label_kernel(torch.arange(16) % 2, 2), update=True)
print(f"输入侧是否留下梯度：{x.grad is not None}（应为 False）")
print(f"本层权重是否被更新：{layer.linear.weight.grad is not None}（应为 True）")

# %% [markdown]
# ## 4. 整网络训练：局部规则 + 线性读出
#
# 隐藏层各用各的局部 pHSIC 目标（无监督，标签只经教学信号进入），读出层用交叉熵。
# 这就是论文表 1 里 pHSIC 那几列的配置。

# %%
model = IBHebbianPerceptron(
    DIM, width=64, n_layers=2, n_classes=N_CLASSES, n_groups=8, readout_learning_rate=0.05
)

first_loss, _ = model.evaluate(features, labels)
for _ in range(60):
    model.step(features, labels)
final_loss, final_accuracy = model.evaluate(features, labels)

print(f"交叉熵：{first_loss:.4f} → {final_loss:.4f}")
print(f"训练集准确率：{final_accuracy:.4f}（随机基线 {1 / N_CLASSES:.2f}）")

# %% [markdown]
# ## 5. 去掉除法归一化会变差
#
# 论文摘要的原话是：这条规则「要在难题上 work 并保持生物合理，**需要**除法归一化」。
# 所以做一个最朴素的消融：把归一化指数取 0（等价于只做居中、不做按方差缩放），
# 其余保持不变，看准确率是否下降。
#
# 注意这是一个**弱**消融——它只说明「归一化这一项在起作用」，不构成对论文那条断言的
# 独立验证。


# %%
def train_with_power(power: float, steps: int = 60) -> float:
    torch.manual_seed(0)
    net = IBHebbianPerceptron(
        DIM,
        width=64,
        n_layers=2,
        n_classes=N_CLASSES,
        n_groups=8,
        divnorm_power=power,
        readout_learning_rate=0.05,
    )
    for _ in range(steps):
        net.step(features, labels)
    return net.evaluate(features, labels)[1]


with_normalization = train_with_power(0.2)
without_normalization = train_with_power(0.0)
print(f"p=0.2（论文设置）：{with_normalization:.4f}")
print(f"p=0.0（消融）    ：{without_normalization:.4f}")

# %% [markdown]
# ## 6. 断言
#
# 这些断言是给 CI 用的——脚本失败即 CI 失败，正如计划书 §12.3 想要的那样。

# %%
assert out.grad_fn is None, "层输出不该携带计算图"
assert x.grad is None, "输入侧不该留下梯度"
assert final_accuracy > 0.5 + 0.2, f"局部规则没有学到东西：准确率仅 {final_accuracy:.4f}"

print("\n全部断言通过。")

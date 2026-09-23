# %% [markdown]
# # 论文最小复现：Frémaux et al. (2010) 的无监督偏差
#
# > Nicolas Frémaux, Henning Sprekeler, Wulfram Gerstner. *Functional Requirements for
# > Reward-Modulated Spike-Timing-Dependent Plasticity.* Journal of Neuroscience
# > 30(40): 13326-13337 (2010).
# > [10.1523/JNEUROSCI.6249-09.2010](https://www.jneurosci.org/content/30/40/13326)
#
# 计划书 §12.3 要求「每篇核心文献对应一个 `examples/paper_<name>.py`」。这篇是 §3.2 那条
# 结论的出处：**R-STDP 存在结构性无监督偏差，唯一的结构性解法是刺激特异性奖励预测**。
#
# 论文原文（Figure 2A 那一段）逐字为：
#
# > Figure 2A shows that success offsets of a magnitude of ∼25% of the SD (σR) of the success
# > signal are sufficient to prevent R-STDP from learning a target spike train in response to a
# > given input spike pattern. Moreover, for a success offset S̄ < −0.4σR (i.e., the average
# > success signal is negative) (Fig. 2A, green points), the performance after learning is even
# > below the performance before learning (Fig. 2A, dotted horizontal line).
#
# ## 本脚本复现了什么、没复现什么
#
# **复现的是那条结论的结构**：
#
# 1. 用一个**全局**奖励滑动平均做基线时，每个刺激各自的成功信号均值 **S̄ₖ 不为零**
#    （这正是"无监督偏差"——偏置项是 `S̄ₖ·⟨e_ij⟩`）；
# 2. 偏移随奖励的逐刺激差异增大而增大，学完的性能**单调下降**；
# 3. 换成**按刺激分开**的奖励预测（§3.2 点名的结构性解法）后，偏移被压回近零，
#    性能在所有偏移幅度下都**不下降**；
# 4. 偏移足够大时,**至少有些种子**学完不如学之前——论文那个 "unlearning" 分支。
#    但它只出现在最差的种子上，**按均值并没有**跌到学之前之下。
#
# **没复现的是那两个具体数值**：论文说 ~25%σR 就足以让 R-STDP 学不会、S̄ < −0.4σR 时
# 会跌到学之前以下。本任务在偏移 **0.26σR** 时准确率仍有 **0.975**（几乎没伤），要一直
# 推到 **0.97σR** 才看到明显退化和个别的 unlearning。本任务（8 个刺激、one-hot 编码、
# 线性 Actor 的查表）比论文的脉冲时序学习任务**宽容得多**，所以数值阈值对不上。
#
# 这条差距写在 `research/rstdp/README.md` 的已知边界里，**不要把本脚本读成对 25% 那个
# 数字的验证**。
#
# **运行位置**：必须在本仓库内运行（`research/` 不是分发包，见 ADR-0001/0004）。

# %%
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import torch  # noqa: E402

from research.rstdp.rstdp import RSTDPActor  # noqa: E402

# %% [markdown]
# ## 任务
#
# 8 个刺激，每个要求一个固定动作（`a*(k) = k % 2`）。**动作分布保持混合**：策略用固定温度
# 的 Boltzmann 采样，两种动作始终有非零概率。
#
# 关键的一步是奖励：`R = b_k + 1{动作正确}`，其中 `b_k` 是**逐刺激的基线偏置**（正负各半、
# 且与正确动作无关）。于是：
#
# * **全局基线**只知道总均值，对每个刺激都减错了基线 → `S̄ₖ ≈ bₖ`，偏移自然出现；
# * **按刺激的预测**对每个刺激各维护一个均值 → `S̄ₖ ≈ 0`。
#
# 偏移是方案的**涌现属性**，不是外面加进去的。扫 `bₖ` 的幅度就得到那条曲线。

# %%
N_STIMULI = 8
N_TRIALS = 6000
LOGIT_SCALE = 3.0
LEARNING_RATE = 0.05
TRACE_DECAY = 0.9
MEAN_ALPHA = 0.02

#: 逐刺激的基线偏置模式：正负各半，且与正确动作无关。
BIAS_PATTERN = [+1.0, -1.0, +1.0, -1.0, -1.0, +1.0, -1.0, +1.0]


def correct_action(k: int) -> int:
    return k % 2


def run(spread: float, *, seed: int, stimulus_specific: bool) -> dict:
    """跑一次，返回偏移（单位 σR）与学之前 / 学之后的贪心准确率。"""
    generator = torch.Generator().manual_seed(seed)
    actor = RSTDPActor(
        n_features=N_STIMULI,
        n_actions=2,
        learning_rate=LEARNING_RATE,
        trace_decay=TRACE_DECAY,
        normalize=True,
        action_sampling="boltzmann",
        logit_scale=LOGIT_SCALE,
        generator=torch.Generator().manual_seed(seed + 1000),
    )
    features = [torch.zeros(N_STIMULI) for _ in range(N_STIMULI)]
    for k, f in enumerate(features):
        f[k] = 1.0

    def accuracy() -> float:
        return (
            sum(
                int(actor.select_action(f, generator=generator, greedy=True) == correct_action(k))
                for k, f in enumerate(features)
            )
            / N_STIMULI
        )

    before = accuracy()

    global_mean = 0.0
    per_stimulus = [0.0] * N_STIMULI
    signals: list[float] = []
    per_stim_signals: list[list[float]] = [[] for _ in range(N_STIMULI)]

    for _ in range(N_TRIALS):
        k = int(torch.randint(N_STIMULI, (1,), generator=generator).item())
        f = features[k]
        action = actor.select_action(f, generator=generator)
        reward = spread * BIAS_PATTERN[k] + (1.0 if action == correct_action(k) else 0.0)

        if stimulus_specific:
            prediction = per_stimulus[k]
            per_stimulus[k] += MEAN_ALPHA * (reward - per_stimulus[k])
        else:
            prediction = global_mean
            global_mean += MEAN_ALPHA * (reward - global_mean)

        signal = reward - prediction
        signals.append(signal)
        per_stim_signals[k].append(signal)
        actor.update(f, action, signal)

    # **起作用的是逐刺激的成功信号均值 S̄ₖ**（偏置项是 S̄ₖ·⟨e_ij⟩），不是全局均值——
    # 全局均值对任何像样的滑动平均都趋于 0，量它等于量了个寂寞。
    sigma = statistics.pstdev(signals)
    offsets = [statistics.fmean(v) for v in per_stim_signals]
    return {
        "before": before,
        "after": accuracy(),
        "offset_ratio": statistics.fmean(abs(o) for o in offsets) / sigma,
    }


# %% [markdown]
# ## 扫一遍

# %%
SPREADS = [0.0, 0.125, 0.5, 2.0]
SEEDS = [0, 1, 2, 3, 4]

print(
    f"{'b_k 幅度':>9} {'偏移/σR':>9} | {'全局基线：学之前 → 学之后':>26} {'最差种子':>9} | "
    f"{'按刺激预测：偏移/σR → 学之后':>28} {'最差种子':>9}"
)
print("-" * 104)
table = []
for spread in SPREADS:
    biased = [run(spread, seed=s, stimulus_specific=False) for s in SEEDS]
    unbiased = [run(spread, seed=s, stimulus_specific=True) for s in SEEDS]
    row = {
        "spread": spread,
        "biased_offset": statistics.fmean(r["offset_ratio"] for r in biased),
        "biased_before": statistics.fmean(r["before"] for r in biased),
        "biased_after": statistics.fmean(r["after"] for r in biased),
        "biased_worst": min(r["after"] for r in biased),
        "unbiased_offset": statistics.fmean(r["offset_ratio"] for r in unbiased),
        "unbiased_after": statistics.fmean(r["after"] for r in unbiased),
        "unbiased_worst": min(r["after"] for r in unbiased),
    }
    table.append(row)
    print(
        f"{spread:>9.3f} {row['biased_offset']:>9.3f} | "
        f"{row['biased_before']:>11.3f} → {row['biased_after']:<11.3f} {row['biased_worst']:>9.3f} | "
        f"{row['unbiased_offset']:>20.3f} → {row['unbiased_after']:<7.3f} {row['unbiased_worst']:>9.3f}"
    )

# %% [markdown]
# ## 断言：这才是本脚本的主张
#
# 四条断言，都在上表里读得出来。**最后一条对应论文那个"unlearning"分支**——它只出现在
# 最差的种子上，所以报的是最小值而不是均值。

# %%
# 1. 全局基线方案的偏移随 b_k 增大而增大——偏差是方案**涌现**出来的，不是外面加进去的。
assert table[-1]["biased_offset"] > table[0]["biased_offset"] + 0.3, table

# 2. 偏移越大，全局基线方案学完的性能越低（方向与论文一致）。
assert table[-1]["biased_after"] < table[0]["biased_after"], table
assert table[-1]["biased_worst"] < table[0]["biased_worst"], table

# 3. 换成按刺激分开的预测后，偏移被压住（远低于全局基线那一侧），
#    而且性能在所有幅度下都保持高位——**这是 §3.2 那条结论的核心**。
assert table[-1]["unbiased_offset"] < table[-1]["biased_offset"] - 0.4, table
for row in table:
    assert row["unbiased_after"] >= 0.85, row
    assert row["unbiased_worst"] >= 0.85, row

# 4. 论文的 "unlearning" 分支：偏移足够大时，**至少有些种子**学完不如学之前。
assert table[-1]["biased_worst"] < table[-1]["biased_before"], table

# %% [markdown]
# ⚠️ **本脚本没有主张的**：上面第 4 条是「有些种子」，不是「平均而言」——`biased_after` 的
# 均值在最大幅度上仍是 0.625，**没有**低于学之前的 0.525。而且那两个**具体数值阈值**
# （~25%σR、−0.4σR）在本任务上都对不出来：偏移到 0.26σR 时准确率还有 0.975。
print(
    "\n⚠️ 论文那两个**数值**阈值（~25%σR 就学不会、S̄ < −0.4σR 跌到学之前之下）"
    "\n   **未被本脚本验证**：本任务在 0.26σR 时准确率仍有 "
    f"{table[1]['biased_after']:.3f}，"
    f"\n   且最大幅度下准确率均值 {table[-1]['biased_after']:.3f} 仍在学之前 "
    f"{table[0]['biased_before']:.3f} 之上——"
    "\n   只有最差的种子（"
    f"{table[-1]['biased_worst']:.3f}）掉到了下面。"
    "\n   详见 research/rstdp/README.md 的已知边界第 11 条。"
)

# P4 草稿 · §3.3 Triviality Analyses: Four Cases

> **状态**：2026-10-24 初稿。素材全部来自已落库的实测，无占位符。
> 位置：`paper/p4/`——与 P2 的 LaTeX 分开，不参与 `paper/main.tex` 的编译检查。
> ⚠️ **章节位置与骨架不一致**：骨架 v0.1 把退化解细节放在**附录 C**，2026-10-24 的裁决把它
> 提到 **§3.3**。本稿按裁决写成正文节；若最终仍放附录，把这些小节降为附录条目即可，
> 内容不用重写。见 [`README.md`](README.md) 的「结构对照」。
> 编号（§3.3 / §5.1 / §5.2）沿用 2026-10-24 定下的 P4 结构。

---

## Editorial note（中文，不进正文）

这一节是 P4 方法学贡献的核心：**四个"看起来成立、其实什么都没测到"的退化解**，每一个都在本仓
真实发生过，且每一个都留下了**能把它挡住的守卫**。写作时保持同一个四段式：

1. 朴素设计长什么样；
2. 它为什么是平凡的（给出可验证的机制，不是"我们觉得"）；
3. 什么守卫把它挡住了（断言 / 门禁 / 对照）；
4. 教训（一句，可迁移到别的基准）。

⚠️ 四案的强度不一样：案 1–2 由早期实验发现，案 3–4 是 2026-10-24 一轮里连续暴露的。
正文要按"发现的顺序"还是"概念的顺序"写，**尚未定**——当前按概念排（从目标到代理到退化）。

---

## §3.3 Triviality Analyses: Four Cases

Every benchmark in this paper was written to answer a question about local learning rules.
Four times, the first version of a measurement turned out to be satisfiable without the
phenomenon it was meant to detect. We report all four, because in each case the *guard* that
now catches the degeneracy is a reusable piece of method, and because a benchmark that only
publishes the versions that worked is a benchmark whose failure modes are unknown.

### 3.3.1 Case one: the arbiter that wins by switching a rule off

**The naive design.** Two plasticity rules propose updates for the same synapses; an arbiter
mixes them with a weight `alpha`; the criterion is that conflict events between the two rules
drop by at least 50%.

**Why it is trivial.** Conflict is measured between the blend and rule A. The blend is
`(1 - alpha) * A + alpha * B`. Setting `alpha = 0` makes the blend *equal* rule A, so the
conflict count is zero, so the criterion is met with room to spare. On a synthetic trace where
B opposes A outright, this reaches a 100% reduction in every seed. The optimum of "minimise
conflict" is not arbitration; it is deletion.

**The guard.** `min_retention` floors `alpha`, and — more importantly — `retention` is reported
*next to* the reduction. A number that says "conflicts down 100%" is now always accompanied by
"and rule B contributes 0%".

**The lesson.** Any objective of the form "reduce a disagreement between X and Y" must state
what happens to Y in the same breath, or the cheapest solution is always to remove Y.

### 3.3.2 Case two: the per-synapse cosine that cannot have an angle

**The naive design.** The plan specifies conflict detection as "cosine similarity of the two
rules' update directions at the same synapse; an angle above 90 degrees is a conflict."

**Why it is trivial.** At a single synapse the "direction" of an update is one scalar, so the
cosine is `+1` or `-1` and the angle is 0 or 180 degrees. The threshold never fires in the
middle; "angle > 90 degrees" collapses to "the signs differ". A threshold with no interior is
not a measurement.

**The guard.** Conflict is computed per *row* (the outgoing weights of one presynaptic neuron),
which is the smallest group that carries a direction. Both conventions exist in the code and are
tested; they are not interchangeable, and the axis argument is documented because it is easy to
read backwards.

**The lesson.** Before a geometric criterion is used, check that the objects it is applied to
have the geometry it assumes.

### 3.3.3 Case three: the evolutionary search with an objective that has no gradient

**The naive design.** A one-parameter evolution strategy tunes `alpha` by hill-climbing on the
number of conflict events.

**Why it is trivial.** The objective is an *integer count*. At `alpha = 0.5` the blend already
conflicts with rule A zero times on every step, so every perturbation also scores zero, and the
search never accepts a move. The adaptive arm therefore produced results **bit-identical** to a
frozen `alpha = 0.5` control, five seeds out of five. This is not "the search found that 0.5 is
optimal"; it is a search with nothing to climb.

**The guard.** Two changes, in order: (i) the objective became continuous; (ii) *every candidate
objective must pass a discrimination check of its own* before it may be used — a pre-registered
rule requiring the candidate to rank outcomes the way the acceptance metric does. Both an
end-to-end objective (Fisher separability, measured on short probes) and a continuous proxy were
put through that check; both failed it, and the adaptive arm was **not run**.

**The lesson.** An objective is a hypothesis about what matters. It has to be tested like one.
The failure is silent — a saturated objective looks exactly like a converged optimum.

### 3.3.4 Case four: the proxy that measures a different construct

**The naive design.** The natural continuous surrogate for "the blend agrees with both rules" is
the mean cosine between the blend and each rule's update.

**Why it is trivial.** Over a grid of `alpha`, this proxy peaks at **0.9** while Fisher
separability peaks at **0.5**; worse, it assigns **the same value** (0.501) to `alpha = 0` and
`alpha = 1`, although the two endpoints differ by 0.14 in Fisher. The proxy is not weak — it is
blind exactly where the decision is made. Alignment and separability are different constructs,
and agreement between two updates says nothing about whether their combination separates classes.

**The guard.** The pre-registered rule from case three applies to proxies too: before a proxy is
allowed to stand in for the acceptance metric, its rank correlation with that metric must clear a
threshold on all seeds. This proxy did not, so it was never enabled.

**The lesson.** A proxy inherits none of the acceptance metric's validity by being smooth. The
cheque has to clear, not just look like a cheque.

---

## 与正文其他节的接口

| 本节 | 连到哪 |
| :--- | :--- |
| 3.3.1 | §5.2 自适应臂的处决（`alpha -> 0` 是 §5.2 里"不可测量"的反面：可测量但无意义） |
| 3.3.3 | §5.2 的任务规格预注册；ADR-0014 的措辞红线 |
| 3.3.4 | §5.1 混合 > 单方（说明为什么"混合有效"不等于"仲裁有效"） |

## 尚未定的写作问题（留给作者）

1. 四案的排序：概念顺序（当前）vs 发现顺序；
2. 案 3 与案 4 是否合成一节「两个候选目标的处决」，因为它们出自同一轮；
3. 是否把"7 小时"的标价写进本节还是留给 §5.2。

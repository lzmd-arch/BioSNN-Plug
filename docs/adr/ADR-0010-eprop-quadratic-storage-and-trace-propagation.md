# ADR-0010：e-prop 的二次存储成立，但 §3.1 的补救措施（上 Trace Propagation）不采纳

- **状态**：已采纳
- **日期**：2026-09-23
- **相关**：计划书 §3.1 关键修正 1、§三、§八 选型表、§十一 风险表；`research/eprop/`；
  [ADR-0007](ADR-0007-td-ltp-critic-provenance.md)（同为引文考证）；
  [docs/references.md](../references.md) 条目 2

## Context

计划书 §3.1「关键修正 1」（`BioSNN-Plug_项目计划书_v6.3.md:161-163`）说两件事：

> e-prop 的资格痕迹按突触存储，空间复杂度随神经元数量**二次增长** [Pes et al., 2025]。
> Traces Propagation（TP）是一种前向、内存高效、可扩展的完全局部学习规则，结合资格痕迹与
> 逐层对比损失，**无需辅助逐层矩阵**……

而 §八 的选型表把它写成了直接的替代关系：「e-prop 存储优化 | Trace Propagation |
**资格痕迹存储从 O(N²) 降至 O(N)**」——同一断言还出现在 §三 的 104/159 行、架构图（86 行）、
363 行、604 行、风险表 496/516 行。

`docs/references.md` 的条目 2 此前记了三条考证，其中第 3 条说：**「原文把该复杂度明确归属给
OSTTP/OSTL，并未单独给出 e-prop 的空间复杂度——计划书把它单独归给 e-prop 属引申。」**

本 ADR 处理两件事：**那条考证本身对不对**，以及**该不该照 §3.1 实现 TP**。

## Decision

### 一、第 3 条考证是错的，予以撤回

从 arXiv:2509.13053v2 的 LaTeXML HTML **逐格**解析 Table 3（不是读 PDF 的文本流，那一层的表格
是错位的），旁证以 v1 与 PDF 交叉核对：

| Model | Update Locking | Weight Transport | Time Local | Space Local | **Space** | Time | Aux |
| :--- | :-: | :-: | :-: | :-: | :--- | :--- | :--- |
| BPTT | ✗ | ✗ | ✗ | ✗ | TLH | TLH² | − |
| **E-prop [2]** | ✗ | ✗ | ✓ | ✗ | **LH²** | **LH²** | **−** |
| E-prop(rnd) [2] | ✗ | ✓ | ✓ | ✗ | **LH²** | LOH | LOH |
| OSTL [8] | ✗ | ✗ | ✓ | ✗ | LH² | LH² | − |
| OSTTP [11] | ✓ | ✓ | ✓ | ✓ | LH² | LOH | LOH |
| TESS [13] | ✓ | ✓ | ✓ | ✓ | LH | LOH | LOH |
| TP (ours) | ✓ | ✓ | ✓ | ✓ | LH | LH | OH |

§1.3.1 更进一步**点名**了 e-prop：先给通式 `ϵ_l^t[i,j] = βϵ_l^{t-1}[i,j] + g(s) f(s)`（Eq. 9，
索引就是逐突触的 `[i,j]`），再写

> For instance, the eligibility trace of **E-prop [2]** defines the presynaptic factor as a
> low-pass filtered version of the spiking activity … and the postsynaptic factor as the surrogate
> derivative of the spike function …

随后给出族属性：「eligibility traces are stored per synapses, leading to a space complexity that
scales quadratically with the number of neurons, i.e., 𝒪(H²L)」。**e-prop 是该段第一个被点名的
实例，`these solutions` 在字面上覆盖它。**

**所以计划书 §3.1 的前件成立、且是原文的直接陈述**（"随神经元数量二次、随层数线性" 与 Table 3
的 `LH²` 逐字吻合）。仓库此前那条考证**对散文成立、对表格不成立**——v2 的 §2.3 那句只点了
OSTTP（v1 那句更明确：说 ETLP 与 OSTTP「based on **E-prop [2]** and OSTL [8]」），当初若只读
散文就会误判。撤回办法见「Consequences」。

### 二、但「上 TP」这条补救措施**不采纳**——三条硬证据

1. **TP 不是 e-prop 的省内存版本，而是换掉空间信用分配的另一条规则。** 论文把两者**分列
   两行、分轴评价**（e-prop：时间局部 ✓ / 空间局部 ✗ / LH² / 无辅助矩阵；TP：全 ✓ / LH / LH / OH），
   没有任何一句说 TP 是 e-prop 的优化。TP 用的是**两条逐神经元**的活动痕迹（Eq. 11/12，索引
   `(batch, neuron)`，递推里没有 ψ_j、没有 ε_v/ε_a、没有 e_ij）+ 一个 one-hot 目标通路 +
   batch×batch 对比损失（Eq. 13-15）+ 更新式（Eq. 18）里**当前时刻**求值的伪导数。
   **采纳它 = 换掉 W2 的学习器**，不是优化它的内存。
2. **它覆盖不到 W2 的神经元模型。** 论文 Table 1/2 里 TP 的**全部**结果都是 **LIF**（SHD 两行
   400/450 也是 LIF；ALIF 那行属于 ETLP）。**W2 的验收配置是 ALIF（β=0.07）。**
3. **在论文自己的头号数据集上，e-prop 比 TP 更好。** N-MNIST：`eProp [2] … 97.90` 对
   `TP (ours) … 97.33 ± 0.06`。而摘要那句「outperforms other fully local learning rules」
   **不包含 e-prop**——因为 e-prop 在 Table 3 里标 `Partial (time)` local，不在「fully local」
   这个比较集内。**读者若把这句读成「本项目选的规则已被证明不如 TP」，方向正好相反。**

（附带一条：TP 需要 batch ≥ 2 才能做对比损失、不能逐样本在线更新；论文自己的 Eq. 25 表明它相对
TESS 的内存优势只在 `O > B` 时成立，而 sMNIST 是 O=10、B=64。但这一条**不作为拒绝理由**——
它拿第三方当靶子，而且 TP 的逐神经元痕迹在这个点上其实只有约 0.125 MiB。**真正的理由是上面三条。**）

## Consequences

- **`docs/references.md` 条目 2 的第 3 条发现要改写**（三语）：从「属引申」改成「前件成立；
  原文由 Table 3 的 `E-prop [2]` 行与 §1.3.1 的点名共同给出；散文（v2 §2.3）只点了 OSTTP，
  容易误读——这大概就是当初判错的原因」。同时收紧另一处措辞：原文**并非**从未出现 `MNIST`
  一词（出现 3 次：摘要的 `NMNIST` 与 §3.1.1 说明 N-MNIST 的来源），准确说法是
  **它从未把 MNIST 用作评测数据集**。
- **`research/eprop/README.md` 的边界第 1 条要改写**（三语）：从「Trace Propagation 尚未实现」
  （读起来像欠账）改成「经考证，§3.1 的补救措施**不适用于本条目**，理由三条」，并说明
  **本实现真正该做的那一条已经做了**（见下）。
- **本实现实际做的**：`traces.py` 里 `epsilon_v` 曾按 `(batch, n_pre, n_post)` 分配，而它的递推
  **不含后突触下标 j**——同一时刻对所有 j 是同一个值。改成 `(batch, n_pre, 1)` 后**逐位等价**
  （`torch.equal=True`、`max|diff|=0`），验收形状下墙钟 **−18.2%**。`epsilon_a` 的系数含
  `(ρ − β·ψ_j)`，**不可因式化**，所以这一段的内存大头由 ALIF 的适应项决定；LIF（β=0）下两者
  都退化成外积，整个迹的状态可降到 O(batch·N)。这条界限写进了代码注释。
- **`research/rstdp/README.md` 的边界**（三语）也把 TP 列为「本阶段尚未实现」——同一处要一起改，
  否则仓库两条线对同一件事的说法不一致。

## Alternatives

- **照计划书实现 TP**：**否决**，理由见 Decision 二。若要实现，它应当作为**另一条学习规则**
  立项（换掉 e-prop 的学习器），而不是作为「e-prop 的存储优化」；且需要先把 W2 的神经元模型
  从 ALIF 换成 LIF，或者自己推导 ALIF 版本的 TP——论文没有给。
- **只改「散文 vs 表格」的措辞、不动实现**：**否决**——因为本实现里**确实**有一块白占的
  O(N²)（`epsilon_v`），改它是零风险的（逐位等价 + −18%）。只改文档等于放着一块免费的改进不做。
- **退到「半局部」以换取更小的存储**：**不适用**——那属于计划书 §十一 的降级路径，与本条目的
  存储问题无关。

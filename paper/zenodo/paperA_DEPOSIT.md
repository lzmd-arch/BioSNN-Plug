# Paper A — Zenodo deposit sheet

**状态：已发布（v1.0，2026-10-31）。** **Version DOI**：10.5281/zenodo.23817507（概念 DOI `10.5281/zenodo.23817506`）。

发布经用户明确要求、通过 Zenodo API 完成；发布后已从**公开 API** 核对：DOI、类型 Preprint、许可 CC-BY-4.0、版本 v1.0、附件 md5 与本地逐字节一致、两篇互挂生效。

## 元数据（逐字段照抄到 Zenodo）

| 字段 | 值 |
| :--- | :--- |
| Title | `Local Learning Rules Under Honest Measurement: A Reproducibility-First Benchmark of e-prop, Hebbian, and R-STDP on Consumer Hardware` |
| Creators | Shen, Jinyao (Independent researcher, <https://github.com/lzmd-arch>) |
| Resource type | Publication → Preprint |
| License | CC-BY-4.0 |
| Version | v1.0 |
| Language | en |
| Keywords | spiking neural networks; local learning rules; e-prop; R-STDP; kernelised IB-Hebbian; reproducibility; pre-registration; negative results; sequential MNIST; CartPole |
| Related: isSupplementedBy | `https://github.com/lzmd-arch/BioSNN-Plug` (software) |
| Related: isSupplementedBy | Paper B, DOI 待填（两篇互挂） |

## Description（Markdown，粘贴到 Zenodo 的 Description 框）

```markdown
We benchmark three purely local learning rules for spiking neural networks — e-prop,
kernelised information-bottleneck Hebbian, and reward-modulated STDP with a TD-LTP critic —
under one measurement protocol, on one consumer GPU, with every criterion pre-registered and
every negative result reported.

Headline results: Hebbian features reach 0.9774 on MNIST over five seeds; e-prop with LIF
neurons reaches 0.8889 on sequential MNIST after 250 epochs, against a matched-budget
backpropagation-through-time control at 0.9630 — the paper's largest measured gap. Neuron type
interacts with learning rate: a non-adaptive LIF neuron beats the adaptive ALIF neuron at every
learning rate swept. The spiking execution layer does not learn CartPole, and sixteen
pre-registered diagnostic rounds trace the failure to credit assignment rather than to the
readout, the sampling temperature, the weight budget or the learning signal's amplitude — after
a fixed input encoding raises the readout's state response thirty-fold and leaves the policy's
angle dependence at zero. An event-driven e-prop whose updates are elementwise identical to
dense e-prop is reported with its cost at 500,000 neurons. A failure library of four
cognitive-layer probes (event-priority updates, neurogenesis, a liquid time constant, and
predictive coding as the third factor) is included with its criteria and mechanisms.

All numbers trace to reproduction records in the linked repository; the LaTeX source is in
`paper/`. Negative results are part of the body, not an appendix.
```

## 要附的文件

| 文件 | 来源 |
| :--- | :--- |
| `paperA_v1.0.pdf` | **已构建**：678,786 字节、sha256 `b67db6eb…`；本地 tectonic 0.17.0 从 commit `bd437cc` 编译（CI 的 `paper-pdf` artifact 受 GitHub egress 配额限制 503，改本地构建）。PDF 不入库——内嵌生成时间，每次重编哈希都变 |
| `paperA_source_v1.0.tar.gz` | `paper/main.tex` + `paper/refs.bib` + `paper/figures/` 的快照（可选，但便于评审复编译） |

## 发布后要做的

1. 把 version DOI 写到本表顶部与 `paper/README.md`；
2. 在两篇的 Related identifiers 里互挂；
3. 把 DOI 填进背书邮件与 EMBER 信；
4. 若之后正文有改动 → **新版本、新 DOI**（不要原地替换文件）。

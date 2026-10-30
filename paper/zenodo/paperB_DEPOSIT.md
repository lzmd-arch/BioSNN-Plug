# Paper B — Zenodo deposit sheet

**状态：已发布（v1.1，2026-10-31）。** **Version DOI**：10.5281/zenodo.23819567（概念 DOI `10.5281/zenodo.23817510`）。

v1.1 只把作者行的中文名（沈锦垚）改为矢量图；v1.0 为 `10.5281/zenodo.23817511`。两篇互挂指向对方的概念 DOI。

## 元数据（逐字段照抄到 Zenodo）

| 字段 | 值 |
| :--- | :--- |
| Title | `Emergence Without Scale: Dynamics and Organisation Tested Against the Substrate Bet in a Spiking Architecture` |
| Creators | Shen, Jinyao (Independent researcher, <https://github.com/lzmd-arch>) |
| Resource type | Publication → Preprint |
| License | CC-BY-4.0 |
| Version | v1.1 |
| Language | en |
| Keywords | spiking neural networks; emergence; local plasticity rules; arbitration; pre-registration; negative results; hybrid LLM architecture; MCP; read/write boundary |
| Related: isSupplementedBy | `https://github.com/lzmd-arch/BioSNN-Plug` (software) |
| Related: references | Paper A, DOI 待填（两篇互挂） |

## Description（Markdown，粘贴到 Zenodo 的 Description 框）

```markdown
Why intelligence emerges is usually asked as a question about scale. This paper tests the
opposite bet — that dynamics (global state variables that bias learning) and organisation (the
arrangement of rules and modules) can carry a system further than substrate size would suggest —
and reports what an instrument built to test it measured.

Three pre-registered readings. First, two local rules sharing one weight: a fixed 50/50 mixture
is measurable (Fisher ratio 0.651 against 0.472 and 0.331), while choosing the mixing weight is
not — no adaptive scheme beats the fixed mixture, and the static optimum is the default value.
Second, a cognitive layer of working memory (0.9269/0.9276), episodic memory (0.805; 0.922 at
K=4), metacognitive gating across several OOD families, a world model with a termination head
(AUROC 1.0 in receding-horizon closed loop), and a state-level end-to-end gate at 18/18. Third,
orchestration by a replaceable language model: real dialogue passes every criterion, while the
same spiking trigger that passes on a synthetic salience stream fails on a real one (AUC 0.732
against 0.7475 for a plain novelty gate, one trigger per conversation; recalibration recovers
hits but not the score's ceiling).

From those failures the paper derives an architectural rule rather than a slogan: writes belong
to the dynamics, the language model may only propose, and the bus is the only door. A released
implementation is attached as software — a read-only, per-call-audited MCP server (four read tools
and five proposal tools, default read-only), with a gate-mediated write path specified against six
pre-registered criteria (specified, not yet built), and a host that enforces read-only at the
storage engine.

All numbers trace to reproduction records in the linked repository; negative results are reported
with the same care as positive ones.
```

## 要附的文件

| 文件 | 来源 |
| :--- | :--- |
| `paperB_v1.1.pdf` | **已构建**：136,674 字节、sha256 `7a144343…`；本地 tectonic 0.17.0 从 commit `bd437cc` 编译（CI 的 `paper-p4-pdf` artifact 受 egress 配额限制 503） |
| `paperB_source_v1.0.tar.gz` | `paper/p4/main.tex` + `paper/p4/refs.bib` 快照（可选） |

## 发布后要做的

与 Paper A 相同：DOI 回填、两篇互挂、把 DOI 填进背书邮件与 EMBER 信。

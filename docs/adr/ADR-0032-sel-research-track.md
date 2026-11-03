# ADR-0032：SEL（脉冲表达层）作为研究项——分阶段、不占主线、不接产品

[English](ADR-0032-sel-research-track.en.md) · [日本語](ADR-0032-sel-research-track.ja.md)

- **状态**：已采纳（P2–P5 的收口见 [ADR-0034](ADR-0034-sel-distillation-archival.md)）
- **日期**：2026-11-03
- **相关**：[ADR-0028](ADR-0028-metacog-gate-threshold-first.md)、[ADR-0029](ADR-0029-research-line-archival.md)、[ADR-0030](ADR-0030-mainline-shift-to-spiking-sos.md)、[ADR-0031](ADR-0031-mainline-to-scale.md)、`docs/sel-applicability.md`、`BioSNN-Plug_项目计划书_v6.3_amend2.md` / `amend3.md`

## Context

用户给了一份预注册 `SEL-001`（脉冲表达层：TI-LIF + CWSSA + 选择性 SSM + 蒸馏 +
在线 R-STDP），想把"换后端模型时语气不变"从 prompt 下沉到脉冲动力学。它引用的
五项工作（SpikeGPT、WTA Spiking Transformer、NeuronSpark、SpikingMamba、NSLLM）
**都真实存在**（2026-11-03 核对 arXiv 编号），但预注册自身的几处前提与仓库现状不符：

1. 教师写成"Phase 2 混合渲染 + Qwen2-0.5B"，而 Phase 2 **不存在**；实际落地的是
   Phase 1（SIR + Ollama 小模型，默认关）。
2. 路径（`axo/agent/responder.py`、`axo/agent/backends/`）在仓里不存在；真实接线点是
   `src/axo/expression/layer.py` 与 `main_window._on_draft_ready`。
3. 它引用的 ADR-0030（脉冲 SoS 主线）已被 **ADR-0031**（主线改规模实验）取代。
4. H3/H5 的阈值（PPL 差 <20%、10k 步 <2 小时、<5 s、<500 MB）没有在本机标定；
   8GB 上 S3 已实测 16k 神经元吃掉 93.6% 显存，S5 饱和。
5. 仓里**没有任何脉冲语言模型或序列生成训练代码**；`AdaptiveMembrane` 只出二值脉冲。

同时，用户要求判断这套方案对**召回**与**门控**是否适用。这两条线已有三次方向一致的
负结果（召回 hit@1 0.024 vs 0.259；基质补全 0.000 vs 0.025；门控 AUROC 0.535），
而 ADR-0028 已把产品门控退化为一个阈值 0.20。

## Decision

1. **SEL 是研究项，不占主线算力。** 主线仍是 ADR-0031 的规模实验；SEL 在
   `research/sel/` 下推进，任何一步过不了自己的预注册就停。
2. **分阶段、先教师后蒸馏。** P0 TI-LIF → P1 召回小实验 → P2 Axo Phase 2（模板 +
   混合渲染，作为教师）→ P3 教师数据（先 300–500 条合成 SIR）→ P4 最小学生蒸馏 →
   P5 在线 R-STDP。P4 过线前不实现 CWSSA/SSM。
3. **TI-LIF 进 `biosnn-bus` 0.6.0**，作为 ALIF 的超集：`D=1, signed=False` 必须与
   `AdaptiveMembrane` 逐位一致（单测）；不写学习规则、不引 torch。
4. **蒸馏信号以实测为准。** Ollama 0.40.11 的原生 `/api/chat` 支持
   `logprobs/top_logprobs`（≤20）；先跑探针，能拿到就做 top-k 截断 KL，拿不到就退化为
   序列级 CE，并在报告里写明那是监督微调，**不叫自蒸馏**。隐藏态经 HTTP 不可得。
5. **召回与门控各做一次小实验，只出报告、不接线。** 判据沿用父预注册；召回用
   `research/spiking_512` 的 gold 语料（线性基线 0.259，有功效），门控基线用词法
   top-1 余弦（0.825）。**已知结果：两条都不过**（召回 TI-LIF 0.012；门控 0.41）。
   这批实验的用途是给 amend1 的"脉冲门控"一个可核对的收场。
6. **不发布权重、不做 model zoo、不上传反馈、不用用户正文训练。** 用户数据只用于
   经授权的离线评测，训练数据先用合成 SIR。

## Alternatives

- **按 SEL-001 原文一次做完（含 CWSSA/SSM、3000 条数据、10k 步蒸馏）**：否决。8GB 上
  16k 神经元已到顶，仓里没有序列生成训练代码；H3/H5 的阈值未标定，做出来也无法判定。
- **不补 Phase 2，直接用 Phase 1 当教师**：否决。用户明确要求先补 Phase 2；而且
  Phase 1 的三项验收（事实保真、0.5B 延迟、三风格离散度）还没跑，拿它当教师等于把
  未验证的东西往下传。
- **把脉冲门控直接接进产品**：否决。ADR-0028 的判据要求 +0.05，现有读数连平都做不到。
- **立即发布 TI-LIF 到 PyPI 并写"可用于表达层"**：否决。先等 P1/G1 与 P2 的读数，
  发布不承担没有证据的主张。

## Consequences

- `biosnn-bus` 升 0.6.0；产品仓继续钉 `biosnn-bus>=0.5.0,<0.6`，直到 Phase 2 稳定后
  再决定是否升依赖。
- 新增 `docs/sel-applicability.md`、`research/spiking_512/prereg_tilif.md`、
  `research/sel/`、`BioSNN-Plug_项目计划书_v6.3_amend3.md`（状态页）。
- 召回/门控的负结果必须与 SEL 的报告分开写；任何文档出现"1.3B 脉冲 LLM"
  "512 神经元门控已验证""发布千岁风格 checkpoint"都按主张漂移处理。
- 反向约束同样成立：**不得用召回/门控的负结果否定 SEL**——任务不同；也不得用 SEL
  的叙事给它们背书。

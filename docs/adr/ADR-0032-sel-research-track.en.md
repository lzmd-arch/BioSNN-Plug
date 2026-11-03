# ADR-0032: SEL (spiking expression layer) as a research track — staged, off-mainline, not wired

[中文](ADR-0032-sel-research-track.md) · [日本語](ADR-0032-sel-research-track.ja.md)

- **Status**: Accepted (P2–P5 are closed by [ADR-0034](ADR-0034-sel-distillation-archival.en.md))
- **Date**: 2026-11-03
- **Related**: [ADR-0028](ADR-0028-metacog-gate-threshold-first.en.md), [ADR-0029](ADR-0029-research-line-archival.en.md), [ADR-0030](ADR-0030-mainline-shift-to-spiking-sos.en.md), [ADR-0031](ADR-0031-mainline-to-scale.en.md), `docs/sel-applicability.md`, `BioSNN-Plug_项目计划书_v6.3_amend2.md` / `amend3.md`

## Context

The user supplied a preregistration, `SEL-001` (spiking expression layer: TI-LIF + CWSSA +
selective SSM + distillation + online R-STDP), to move "the tone does not change when the
backend model changes" from a prompt into spiking dynamics. Its five cited works (SpikeGPT,
WTA Spiking Transformer, NeuronSpark, SpikingMamba, NSLLM) **do exist** (arXiv IDs checked
2026-11-03), but several of its premises do not match this repository:

1. The teacher is described as "Phase 2 hybrid rendering + Qwen2-0.5B", yet Phase 2 **does
   not exist**; only Phase 1 (SIR + local Ollama model, off by default) does.
2. Paths such as `axo/agent/responder.py` and `axo/agent/backends/` do not exist; the real
   wiring points are `src/axo/expression/layer.py` and `main_window._on_draft_ready`.
3. Its referenced ADR-0030 (spiking SoS mainline) has been superseded by **ADR-0031**
   (mainline moved to the scale experiments).
4. The H3/H5 thresholds (PPL gap <20%, 10k steps <2 h, <5 s, <500 MB) were never calibrated
   here; on 8 GB, S3 measured 16k neurons consuming 93.6% of memory, and S5 saturated.
5. There is **no spiking language-model or sequence-generation training code** in the repo;
   `AdaptiveMembrane` emits only binary spikes.

The same request also asked whether this scheme applies to **recall** and **gating**. Both
lines already have three consistent negative results (recall hit@1 0.024 vs 0.259; substrate
completion 0.000 vs 0.025; gating AUROC 0.535), and ADR-0028 reduced the product gate to a
single threshold of 0.20.

## Decision

1. **SEL is a research track and does not consume mainline compute.** The mainline remains
   the scale experiments of ADR-0031; SEL proceeds under `research/sel/` and stops at any
   stage whose preregistered gate fails.
2. **Staged, teacher first.** P0 TI-LIF → P1 recall mini-experiment → P2 Axo Phase 2
   (templates + hybrid rendering, used as the teacher) → P3 teacher data (300–500 synthetic
   SIRs first) → P4 minimal student distillation → P5 online R-STDP. CWSSA/SSM are not
   implemented before P4 passes.
3. **TI-LIF ships in `biosnn-bus` 0.6.0** as a superset of ALIF: `D=1, signed=False` must be
   bit-for-bit identical to `AdaptiveMembrane` (unit test); no learning rules, no torch.
4. **The distillation signal is decided by measurement.** Ollama 0.40.11's native
   `/api/chat` supports `logprobs/top_logprobs` (≤20); run the probe first. If available,
   use truncated top-k KL; otherwise fall back to sequence-level CE and call it supervised
   fine-tuning, **not self-distillation**. Hidden states are unavailable over HTTP.
5. **Recall and gating each get one small experiment, reported only, never wired.** The
   criteria follow the parent preregistration; recall uses the `research/spiking_512` gold
   corpus (lexical baseline 0.259, so the test has power), gating uses the lexical top-1
   cosine baseline (0.825). **Known result: both fail** (recall TI-LIF 0.012; gating 0.41).
   Their purpose is to give amend1's "spiking gate" a checkable closure.
6. **No weight release, no model zoo, no uploaded feedback, no training on user text.**
   User data is only used for authorized offline evaluation; training starts from synthetic
   SIRs.

## Alternatives

- **Do SEL-001 literally in one go (CWSSA/SSM, 3000 pairs, 10k steps)**: rejected. 16k
  neurons already hit the 8 GB ceiling, the repo has no sequence-generation training code,
  and the H3/H5 thresholds are uncalibrated.
- **Skip Phase 2 and use Phase 1 as the teacher**: rejected. The user explicitly asked for
  Phase 2 first, and Phase 1's three pending acceptance items are still unrun.
- **Wire the spiking gate into the product**: rejected. ADR-0028 requires +0.05; current
  numbers cannot even tie.
- **Publish TI-LIF to PyPI immediately with an "expression layer" claim**: rejected. Wait
  for P1/G1 and P2 numbers; a release must not carry claims without evidence.

## Consequences

- `biosnn-bus` becomes 0.6.0; the product repo keeps pinning
  `biosnn-bus>=0.5.0,<0.6` until Phase 2 stabilizes.
- New documents: `docs/sel-applicability.md`, `research/spiking_512/prereg_tilif.md`,
  `research/sel/`, `BioSNN-Plug_项目计划书_v6.3_amend3.md`.
- Recall/gating negative results must be reported separately from SEL; any document claiming
  "1.3B spiking LLM", "512-neuron gate validated", or "Qiansui style checkpoint released"
  is treated as claim drift.
- The converse also holds: **the recall/gating negatives must not be used to dismiss SEL** —
  the tasks differ — and SEL's narrative must not be used to endorse them.

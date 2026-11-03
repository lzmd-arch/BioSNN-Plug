# SEL-001 research line: current status

[中文](README.md) · [日本語](README.ja.md)

**Teacher data (P3)**: passed (fourth round). The data lives in `D:\AxoData\sel\pairs\v1_amend3\`
(not in the repository); `manifest.json` records sha256 `fedfa34a…`.

| Round | Teacher | Requests | Styled | After dedup | Verdict |
| :--- | :--- | ---: | ---: | ---: | :--- |
| v1 | 0.5B | 400 | 76.5% | 101 | D1 fails (echo + refusal-family collapse) |
| v1_amend1 | 0.5B | 400 | 76.7% | 272 | D1 fails |
| v1_amend2 | 1.5B | 400 | 76.7% | 257 | D1 fails |
| **v1_amend3** | **1.5B** | **600** | **76.7%** | **391** | **D1-D4 all pass** |

Final criteria: D1 391 >= 300 OK; D2 76.7% >= 70% OK; D3 non-refusal unique rate 0.925 and the
largest single share 0.26% OK; D4 split 301/44/46 with zero overlap OK; refusal share 4.9%
(cap 5%).

Pre-registration and its three amendments: `prereg_p3.md`, `prereg_p3_amend1.md`,
`prereg_p3_amend2.md`, `prereg_p3_amend3.md`. Generator: `gen_data.py` (it writes refusals to the
repository root).

## Next

Before P4, run the **logits probe** (`probe_logprobs.py`): how much of the teacher's
`top_logprobs` is reachable and what the coverage looks like. If it is unavailable, fall back to
sequence-level CE and state plainly that this is "supervised fine-tuning".

## P4 student (pilot, 1 seed x 1500 steps; not the final verdict)

| Arm | test top-1 | test CE | seconds |
| :--- | ---: | ---: | ---: |
| bigram (denominator) | 0.6457 | -- | -- |
| GRU(256) | **0.8787** | 0.5019 | 14.4 |
| ALIF(512) | 0.8641 | **0.4227** | 105.5 |
| TI-LIF(512, D=3) | 0.8663 | 0.5003 | 115.1 |

- P4-1 (>= 0.7457): both spiking arms pass; P4-2 (>= GRU - 5pp = 0.8287): both pass;
- **P4-3 fails**: TI-LIF CE is **1.18x** ALIF (the criterion asks <= 0.90x) -- ternary spikes
  do not lower cross-entropy on this character task;
- P4-4 is N/A (see `prereg_p4_amend1.md`); P4-5 not done yet;
- This is a **pilot**: the preregistration asks for 5 seeds x 10k steps; the verdict waits
  for the full run.

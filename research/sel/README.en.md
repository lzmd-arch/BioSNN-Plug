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

## P4 student (5 seeds x 10k steps; final verdict)

| Arm | test top-1 (median [range]) | test CE (median [range]) | median seconds |
| :--- | ---: | ---: | ---: |
| bigram (denominator) | 0.6457 | -- | -- |
| GRU(256) | **0.8831** [0.8795, 0.8846] | 0.5203 [0.4686, 0.5540] | 83 |
| ALIF(512) | 0.8685 [0.8634, 0.8729] | **0.4594** [0.4304, 0.5518] | 620 |
| TI-LIF(512, D=3) | 0.8641 [0.8532, 0.8685] | 0.5274 [0.4987, 0.8482] | 946 |

- P4-1 (>= 0.7457): pass (0.8641);
- P4-2 (>= GRU - 5pp = 0.8331): pass;
- **P4-3 fails**: TI-LIF CE is **1.148x** ALIF (criterion <= 0.90x). In 4 of the 5
  seeds TI-LIF has the higher CE than ALIF for the same seed (only seed 3 is lower);
  ternary spikes do not lower cross-entropy on this character task.
- P4-4 is N/A (see `prereg_p4_amend1.md`); P4-5 not done; P4-6 passes (median 946 s,
  peak 98.4 MB, far below 2 h / 7.5 GB).
- **Per the preregistered "stop if any criterion fails": P4-3 fails, so no
  CWSSA/SSM and no KD follow-up; P4 closes as a negative result.**
- The earlier 1-seed x 1500-step pilot is superseded by this 5-seed table.

# 512 ALIF neurons: a feasibility test for recall and gating (**negative result**)

[中文](README.md) · [日本語](README.ja.md)

Criteria are in [prereg.md](prereg.md) (frozen before the run). Script: [run_512.py](run_512.py).

## Result (gold corpus, 3 projection seeds, top-50 prefilter)

| Metric | 512 population | Baseline (assoc-code cosine) | Delta |
| :--- | ---: | ---: | ---: |
| hit@1 | 0.024 | **0.259** | **-0.235** |
| hit@5 | 0.094 | **0.424** | **-0.329** |
| gating AUROC | 0.535 [0.518, 0.555] | +0.05 criterion not met | -- |

**Both criteria fail** (A1 asked for >= +0.05; the result is -0.235). Recorded as a negative
result; no tuning afterwards.

## Why it failed (written after the result; kept apart from the criteria)

- The associative code is a 64-bit sparse code; cosine measures "how many bits are shared"
  directly, while 512 ALIF neurons compute a coincidence bonus that threshold and refractory
  flatten out. The gold `name_number` class differs by a single number, and the population
  cannot tell those neighbours apart.
- In short: **the linear dot product is already a strong baseline here**, and squeezing the same
  information into a firing rate only loses resolution.

## Conclusion for both uses

- **Recall**: do not wire it. A 0.235 gap is not something a gain parameter can close.
- **Gating**: AUROC 0.535 is near chance; the existing single threshold (0.20) is calibrated on
  the real distribution, so there is no reason to swap in a worse, more expensive implementation.

## Reproduce

```bash
uv run python research/spiking_512/run_512.py --seeds 3
```

## The encoder changed twice; both changes are in the script comments

(1) whole-sentence hashing -> shared words did not overlap (baseline hit@1 = 1/85);
(2) `re.split(r"\W+")` does not work for Chinese (the whole sentence became one token) ->
character bigrams. Both changes touched the **encoder**, never the criteria; only then did the
baseline return to a credible 0.259.

## TI-LIF revision (2026-11-03, 10 projection seeds; prereg in [prereg_tilif.md](prereg_tilif.md))

Signed multi-level spikes (TI-LIF, D=1/2) and K-window averaging (K=5, **five independent
projections**) each get one arm:

| Arm | hit@1 | hit@5 | gating AUROC |
| :--- | ---: | ---: | ---: |
| ALIF K=1 | 0.012 | 0.082 | 0.410 |
| TI-LIF D=1 K=1 | 0.012 | 0.047 | 0.406 |
| TI-LIF D=2 K=1 | 0.012 | 0.047 | 0.410 |
| ALIF K=5 | **0.035** | **0.100** | 0.412 |
| TI-LIF D=2 K=5 | 0.012 | 0.059 | 0.369 |
| Lexical baseline (assoc-code cosine) | **0.259** | **0.424** | **0.825** |

Criteria A1', A1'', A2', B1 and B1' **all fail**. The K-window positive control (R0b)
passes: ALIF goes from 0.012 to 0.035 hit@1 (+2.3pp), so the K-window really introduces
variance; TI-LIF does not benefit from it. The gating baseline here is the lexical top-1
cosine (AUROC 0.825), not 1.000 -- the spiking arms are still 0.41 behind.

**Conclusion unchanged**: TI-LIF does not rescue recall or gating; neither is wired.
Raw JSON under `sweep_results/spiking_512_tilif/`.

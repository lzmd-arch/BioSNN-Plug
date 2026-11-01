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

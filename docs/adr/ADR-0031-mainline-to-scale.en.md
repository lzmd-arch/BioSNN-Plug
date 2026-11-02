# ADR-0031: The main line moves from spiking SoS to the scale experiment

[中文](ADR-0031-mainline-to-scale.md) · [日本語](ADR-0031-mainline-to-scale.ja.md)

- **Status**: Accepted
- **Date**: 2026-11-03
- **Related**: [ADR-0029](ADR-0029-research-line-archival.en.md), [ADR-0030](ADR-0030-mainline-shift-to-spiking-sos.en.md), `research/sos/`, `benchmarks/maze/`

## Context

ADR-0030 made spiking SoS the main line on the strength of SoS's central claim: **showing the search
process teaches search better than showing optimal outcomes** (H1' below). It listed three
prerequisites and forbade training until they landed.

All three landed (P1 in `sweep_results/sos_e1_v4/`, P2/P3 implemented across E2-E7), and then H1' was
tested **twice, independently**:

| Setting | Model | Corpus | `D_search - D_opt` |
| :--- | :--- | :--- | :--- |
| E6 | tabular softmax (displacement + walls + heading) | 100 mazes | **-0.015 [-0.045, +0.015]** |
| E7 | GRU(128) | 2000 mazes | **-0.005 [-0.050, +0.050]** |

Both intervals straddle zero, and neither point estimate is positive. In E7 both arms solve the maze
at **0.90** -- the task is solved and the difference still cannot be measured.

The line did leave three things that stand: the maze benchmark (23 difficulty levels), the world-model
uncertainty (p1_ready: rho 0.842, coverage 5/5), and the "**state/memory is the bottleneck**" ladder
(0.14 -> 0.39 -> 0.74 -> 0.93).

But as a **main line** its central claim has no supporting evidence. On 2026-11-03 the user decided:
the main line moves to the **scale experiment**.

## Decision

1. **The main line moves to the scale experiment**: on the event-driven e-prop path, push the
   recurrent layer to **5k-8k neurons** and answer one concrete question -- **how does accuracy grow
   with size at this scale, and what does it cost (memory / wall clock)?**
2. **SoS is demoted to a research item and no longer takes main-line compute.** Its three assets stay
   citable: the maze benchmark (`benchmarks/maze/`), the world-model uncertainty (E1), and the state
   ladder.
3. **The scale criteria must be on disk before running**: no "bigger is better" narrative -- report an
   **accuracy-versus-size curve under a fixed budget** together with the resource curve.

## Known hard constraints (written down so we do not re-walk them)

- The **dense path tops out near 1.35k recurrent neurons** (plan v6.3 resource scan, log-log slope 0.863);
- The **event-driven path has already run 10k**: 320k sparse weights, one update 0.10 s, a full
  1 epoch over 60k samples in **40 min / acc 0.697** (single epoch, not an accuracy claim);
- A 500k **single-batch update** takes 5.27 s, which rules out epoch training by extrapolation.

So 5k-8k is **not a "will it fit" question, it is a "is it worth it once it fits" question** -- which
is exactly what we will measure.

## Re-evaluation conditions

| Condition | Conclusion |
| :--- | :--- |
| 5k-8k beats 1k-2k by **<= 2 percentage points** | Scale is not the bottleneck here; stop widening and change axis |
| Epoch wall clock exceeds **4 hours** | Not iterable on an 8 GB consumer card; archive as unusable |
| Event-driven is **clearly below** dense at the same size (if dense fits) | The sparse implementation is flawed; fix it before talking scale |

## Consequences

- Compute moves from SoS to the scale sweep;
- SoS's assets are not wasted: the maze benchmark and the state-ladder conclusion are already frozen,
  and H1's null result is recorded here so nobody has to re-run it;
- Cost: the scale experiment will **probably produce a flat curve** -- W2 already reaches 0.8889 at
  256 neurons, and whether widening helps is unknown. Stated up front so a flat curve is not later
  explained away as "not tuned yet".

## Alternatives

- **Keep investing in SoS** (transformer + Countdown reproduction): highest cost, and twice unsupported.
  Rejected.
- **Return to Paper C methodology**: its prerequisite (40 annotations) is not in my hands, and a 0.439
  evidence-coverage means the end-to-end comparison mostly measures the answerer. Not suitable as a
  main line.
- **Set no main line and enter maintenance**: the user chose to push scale instead. Rejected.

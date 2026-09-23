# Reference list

[中文](references.md) · [日本語](references.ja.md)

> **This file is the single source of truth for citation information.**
>
> The `[n]` numbers in the body of the project plan correspond to the "No." column of
> this table. A new citation must be registered here, or neither
> `scripts/check_references.py` nor lychee in CI will see it.
>
> Division of labour: **this table's structure** (consecutive numbers, URL format, every
> entry carrying a verification status) is checked at commit time by
> [`scripts/check_references.py`](../scripts/check_references.py); **whether the links are
> alive** is checked by lychee in CI. Neither of them checks whether the cited content is
> correct — that is what the "verification status" column has to reflect honestly.

## Errors that have already happened

§12.3 of the project plan is quite specific about what the "citation link liveness check"
is for: it is the entry point for intercepting **"Khacef attribution"-type errors**. That
error really did happen in this repository — the author of reference [8] was miscredited as
"Khacef et al.", and it was corrected to Hajizada et al. only in v6.1.

## Verification status

| Status | Meaning | What to do |
| :--- | :--- | :--- |
| `verified` | Verified entry by entry: the work exists, the metadata is correct, and the **specific numbers** the body attributes to it can be found in the original | Nothing |
| `partial` | The work exists and the metadata is correct, but some of the attributed numbers were not confirmed in the original | Verify those numbers |
| `metadata-error` | The work exists, but the registered author / year / volume-issue-pages / article number is wrong | **Fix the bibliography** |
| `unverified` | Not verified yet. **Must be upgraded before citing it externally** | Go and verify it |

## The list

| No. | Work | URL | Verification status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625. | https://www.nature.com/articles/s41467-020-17236-y | verified | Author, volume and article number all match. All three attributions confirmed. ⚠️ Two statements should be made more precise against the original: Equation (28) in the original is ΔW_ji = −η Σ_t L_j^t ē_ji^t, **with a minus sign and summed over time**, and the learning signal is indexed by the postsynaptic neuron j (the project plan writes it as L_i(t) and in instantaneous form). |
| 2 | Pes, L., Yin, B., Stuijk, S., & Corradi, F. (2025). Traces Propagation: Memory-Efficient and Scalable Forward-Only Learning in Spiking Neural Networks. *arXiv preprint* arXiv:2509.13053. | https://arxiv.org/abs/2509.13053 | metadata-error | ⚠️ **Wrong dataset name**: the body of the project plan says "on MNIST and SHD"; the original says **N-MNIST** (an event-camera dataset) — the two are not the same thing; the original **never uses MNIST as an evaluation dataset** (the token appears 3 times: the abstract's `NMNIST`, and §3.1.1 explaining where N-MNIST comes from). ⚠️ **Incomplete provenance**: this paper has been published in *Neuromorphic Computing and Engineering* **6(1):014002 (2026)**, DOI 10.1088/2634-4386/ae2ef9; the journal version should be added. Also: **the item "per-synapse storage leads to O(N²)" holds, and the original does give e-prop's space complexity separately** — Table 3 has its own `E-prop [2]` row with Space Complexity = `LH²`, and §1.3.1 names e-prop as an instance of that family property. **The "extrapolation" finding previously recorded here was wrong and is retracted** (it read only the prose: v2's §2.3 sentence names only OSTTP; v1's is more explicit, saying ETLP and OSTTP are "based on E-prop [2] and OSTL [8]"). Verdict and cell-by-cell evidence: `docs/adr/ADR-0010`. |
| 3 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2010). Functional Requirements for Reward-Modulated Spike-Timing-Dependent Plasticity. *Journal of Neuroscience*, 30(40), 13326-13337. | https://www.jneurosci.org/content/30/40/13326 | verified | Author, volume, issue and pages all match, and there is no attribution error. 4 of the 5 attributions confirmed. ⚠️ The URL is fine in a browser but returns **403** (Cloudflare) to scripted clients — lychee in CI has been given a pass with `--accept ...403`; this is expected behaviour, not a dead link. The attributed original text is "±25% of the SD (σR)" (both directions). |
| 4 | Pogodin, R., & Latham, P. E. (2020). Kernelized information bottleneck leads to biologically plausible 3-factor Hebbian learning in deep networks. *Advances in Neural Information Processing Systems*, 33. | https://proceedings.nips.cc/paper/2020/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html | partial | Three-factor structure, the third factor needs no top-down transmission, and divisive normalization is required — all three confirmed. ⚠️ **"Does not suffer a deep representation bottleneck" was not found in the original** (`not_found`): what the original says is that this rule family **avoids** the representation-bottleneck problem of deep networks, which is a statement of a different strength from "does not suffer"; rewrite it to follow the original. |
| 5 | Confavreux, B., Agnes, E. J., Zenke, F., Sprekeler, H., & Vogels, T. P. (2025). Balancing complexity, performance and plausibility to meta learn plasticity rules in recurrent spiking networks. *PLoS Computational Biology*, 21(4), e1012910. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910 | verified | Metadata matches. ES meta-learning, complex rules starting to fail, and the difficulty of designing the loss function a priori — all three confirmed. "Robustly stabilises all four synapse types" is `close` (the original's robustness conclusion is conditionally qualified, not unconditionally true). |
| 6 | Shen, J., Xie, Y., Xu, Q., Pan, G., Tang, H., & Chen, B. (2025). Spiking Neural Networks with Temporal Attention-Guided Adaptive Fusion for imbalanced Multi-modal Learning. *Proceedings of the 33rd ACM International Conference on Multimedia*. | https://dl.acm.org/doi/10.1145/3746027.3755622 | partial | TAAF's dynamic importance allocation and temporally heterogeneous hierarchical integration — both confirmed. ⚠️ **"Provides dual temporal-fusion and semantic-fusion channels" was not found in the original** (`not_found`). The "dual-channel fusion" of §2.2 in the project plan is **this project's own architectural design** and should not be attributed to TAAF. ACM DL returns 403 to scripted clients; lychee has been given a pass. |
| 7 | Hu, K., Wen, L., Zhang, T., & Zhang, H. (2026). PS-SNN: pattern separation learning for expandable spiking neural networks in class-incremental learning. *Scientific Reports*, 16, Article 12653. | https://www.nature.com/articles/s41598-026-42970-6 | metadata-error | ⚠️ **Wrong article number**: the project plan writes "Article 42970"; authoritative sources (the Nature publication page and Crossref) both give **Article number 12653**. 42970 is the tail of the DOI, not the article number. Both attributions — the 76.42% incremental accuracy and the orthogonal class centers — are confirmed. |
| 8 | Hajizada, E., Rager, D., Shea, T., Campos-Macias, L., Wild, A., Hüllermeier, E., Sandamirskaya, Y., & Davies, M. (2026). Online Continual Learning on Intel Loihi 2 via a Co-designed Spiking Neural Network. *arXiv preprint* arXiv:2511.01553. | https://arxiv.org/abs/2511.01553 | metadata-error | The attribution is confirmed correct (the correction of "Khacef et al." to Hajizada et al. in v6.1 is right and should be kept). ⚠️ **Wrong year**: the arXiv number 2511 means November 2025, with the first submission on 2025-11-03; 2026 is only the year of the v2 revision. ⚠️ **Version and figures mismatched**: the project plan cites the title of v2 but uses the figures of v1 (70× / 23.2ms / 5,600× / 281mJ) — v2 has changed them to 113× / 37.3ms / 6,600× / 333mJ. Take one of the two; do not mix them. |
| 9 | Savage, W. (2026). EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture. *arXiv preprint* arXiv:2604.12167. | https://arxiv.org/abs/2604.12167 | verified | **The entire evidence base of §4 of the project plan, and also the highest-priority verification target.** Metadata matches, and all 14 attributed numbers were **confirmed one by one**: 82.2% / 83.8% discriminability, s=0.14, σ=0.1 and 0.9Hz, first trigger on the 7th conversation turn, 52 messages over 3 days with 23 calls (1 reach_out + 22 journal), 3× impulse threshold and 3 times in 5 minutes, 24 lateral spikes in 15 minutes, synaptic connections 0→10,843→53,992→201,394, 1.6% decay, 64 nodes and 124 edges, Claude Sonnet 4.6, 220K neurons / RTX 5070 Ti and RTX 4060 Ti. Not one of them fell through. |
| 10 | tfatykhov. (2026). MEMBRAIN: Neuromorphic Memory Bridge for LLM Agents. *GitHub Repository*. | https://github.com/tfatykhov/membrain | metadata-error | ⚠️ **Wrong author field**: the project plan treats the project name "MEMBRAIN" as the author/institution; this is a personal repository, and the author is the GitHub account **tfatykhov**. FlyHash encoding (1536→20,000 dimensions, int8 random projection + WTA, about 30MB), Nengo + Voja, and sleep-period noise consolidation — all four attributions confirmed. |
| 11 | christophejlegros-lgtm. (2026). ASTRA: Unified Research Lab + MCP Server. *GitHub Repository*. | https://github.com/christophejlegros-lgtm/ASTRA-Unified-ResearchLab-MCP-v2.7 | metadata-error | ⚠️ **Wrong author field**: as above, the project name "ASTRA" is treated as the author; it is actually the GitHub account **christophejlegros-lgtm**, not an organization. ⚠️ The project plan says it "verified that an SNN engine can be exposed to clients such as Claude Desktop through an MCP server" — the repository does do this, but "verified as usable" is a stronger claim than "provides that interface"; rewrite it according to the actual strength of the evidence. |
| 12 | Frémaux, N., Sprekeler, H., & Gerstner, W. (2013). Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons. *PLoS Computational Biology*, 9(4), e1003024. | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024 | verified | **This is where the P0 to-do (TD-LTP provenance verification) is settled**; see the next section. Verbatim from the original: "Because it has, roughly, the form of 'TD error signal × Hebbian LTP', we call this learning rule TD-LTP." (the Critic learning section, corresponding to Eq. 17 and to the Figure 2A caption "TD-LTP is the learning rule given in Eq. 17.") The paper is open access and the full text can be checked. **Not yet introduced into the project plan**; it needs to be added in v6.3. |
| 13 | Tihomirov, Y., Rybka, R., Serenko, A., & Sboev, A. (2025). Combination of reward-modulated spike-timing dependent plasticity and temporal difference long-term potentiation in actor-critic spiking neural network. *Cognitive Systems Research*, 90, 101334. | https://doi.org/10.1016/j.cogsys.2025.101334 | verified | A follow-up line of work applying TD-LTP to spiking actor-critic. Conference version: Tihomirov, Rybka, Serenko & Sboev (2024), BICA 2024, *Studies in Computational Intelligence*, pp. 411-415, DOI 10.1007/978-3-031-76516-2_41. There is also open-access prior work from the same group, Vlasov et al. (2024), *Moscow University Physics Bulletin* 79(S2):S944-S952, DOI 10.3103/S0027134924702400, which contains the CartPole details. **Not yet introduced into the project plan**. |
| 14 | fangwei123456. (2026). SpikingJelly: An open-source deep learning framework for spiking neural networks based on PyTorch. *GitHub Repository*. | https://github.com/fangwei123456/spikingjelly | verified | **License fact check**: §12.1 of the project plan says "same license as SpikingJelly", but that is **not** the case — SpikingJelly is not Apache-2.0. Its LICENSE is the **Open-Intelligence Open Source License 1.0** (OIOSL); the GitHub API's license field is `NOASSERTION`; and the PyPI stable release 0.0.0.0.14 carries the classifier `License :: Other/Proprietary License`. Commercial use or redistribution requires prior public disclosure to AITISA (aitisa.org.cn); the patent grant is a voluntary declaration conditioned on a patent pool / FRAND. In addition, 2.0.0rc1 requires Python >= 3.11. The ruling and trade-offs are in `docs/adr/ADR-0008` |
| 15 | Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2019–2020). eligibility_propagation: the official implementation of paper [1]. *GitHub Repository*. | https://github.com/IGITUGraz/eligibility_propagation | verified | **License check**: the LICENSE text is standard **BSD-3-Clause** (source retention, binary reproduction, no endorsement); the GitHub API reports `NOASSERTION` only because the copyright header has a non-standard format, not because the license is unclear. Compatible with Apache-2.0. **Reading reference and cross-validation only, not a dependency** (handled the same way as [14] SpikingJelly). What was checked: its dynamics (`EligALIF.__call__` — no `(1−α)` factor on the input current, reset term uses the base threshold while the spike criterion uses the adapted one), its implementation of the Eq. (25) eligibility traces, and its `numerical_verification_eprop_factorization_vs_BPTT.py` |

| 16 | BEKO2210. (2026). Javis: An associative SNN memory co-processor for LLM agents. *GitHub Repository*. | https://github.com/BEKO2210/Javis | verified | **A same-family open-source precedent (read-only reference; not a dependency)**. Positioning: it turns an LLM agent's memory into **cell assemblies** inside a spiking network; a query acts as a partial cue and pattern completion reactivates the assemblies, so only a few decoded concepts are handed to the LLM (the author claims 35–45% fewer tokens than naive RAG). All learning rules are STDP-family (pair / iSTDP / triplet / reward-modulated STDP / BCM / SFA / structural plasticity); **no e-prop, no backpropagation**; pure Rust, no SNN framework at all. ⚠️ **Its licence is PolyForm Noncommercial 1.0.0** — not OSI-approved, not an SPDX standard, with a research-use addendum banning production deployment and third-party services: **no code from it may be merged into this repository**; treat it exactly like [15] (read-only, conclusions only, never a dependency). ⚠️ **Name collision**: other projects called Javis include the Julia animation library `JuliaAnimators/Javis.jl` (**MIT**), `JavisVerse/JavisGPT` and `JavisVerse/JavisDiT` — a licence audit can easily attribute MIT to the wrong one. ⚠️ **No paper and no DOI**; 0 stars, dormant since 2026-05; every number in its README (100% self-recall, 35–45% token reduction, ≈2% associative recall, ≈50-concept capacity) comes from the author's own small-corpus benchmark with **no third-party reproduction**, so it can only be cited as a precedent, never as external evidence |
| 17 | Korcsák-Gorzo, A., Espinoza Valverde, J. A., Stapmanns, J., Plesser, H. E., Dahmen, D., Bolten, M., van Albada, S. J., & Diesmann, M. (2025). Event-driven eligibility propagation in large sparse networks: efficiency shaped by biological realism. *arXiv preprint* arXiv:2511.21674. | https://arxiv.org/abs/2511.21674 | verified | **The event-driven implementation of e-prop** (not "event priority"): it replaces "synchronous updates at every time step" with "update only when a spike event arrives at that synapse", integrates it into NEST, reproduces the original performance on pattern generation / evidence accumulation / N-MNIST, and scales (weak and strong) to **2 million neurons**. To honour **strict locality** it also removed a violation in the original: the eligibility filter depended on the output neuron's time constant (verbatim: "for synapses to compute their weight updates, they must know the time constant of the output neuron, which violates the principle of locality") -- the single most useful sentence here for a "purely local" narrative. ⚠️ `priorit*` has **zero** hits in the paper; there is **no** "event priority" mechanism in it. ⚠️ Spelling: arXiv and publishers write `Korcsak-Gorzo` (no accents) while the literature generally uses Korcsák-Gorzo; ⚠️ name collision: Katherine Korcsak-Gorzo in particle physics is a different person. ⚠️ The author's PhD thesis (RWTH Aachen 2025, D 82, CC BY 4.0) contains this work as its chapter 3 and says "Submitted to Nature Computational Science", but as of 2026-09-23 no journal version was found |
| 18 | Millidge, B. (2025). Generalizing E-prop to Deep Networks. *arXiv preprint* arXiv:2512.24506. | https://arxiv.org/abs/2512.24506 | verified | **A purely mathematical note with no experiments at all**: it extends the eligibility-trace recursion from a single recurrent layer to arbitrarily deep networks and even arbitrary DAGs, arguing that complexity stays linear in depth. The author writes in the Discussion: "we have performed no experiments demonstrating that good credit assignment across depth works in practice" -- so it **must not be used to support any performance number**. Its self-declared limitations are in fact counter-evidence for this project's online-learning claim: e-prop "does not actually perform online weight updates" (it waits until the end of an episode) and storing one set of traces per parameter group becomes "quickly unmanageable" at depth. ⚠️ Two spellings coexist: the PDF and arXiv HTML use `Generalizing`, while arXiv metadata / abs page / OpenAlex use `Generalising`; this table follows the paper itself and notes the difference. ⚠️ It has **no author relationship** to Predictive E-prop (Millidge is not among its authors, and that paper does not cite this note) |
| 19 | Noè, D., Yamamoto, H., Katori, Y., & Sato, S. (2026). Predictive E-prop: A biologically inspired approach to train predictive coding-based recurrent spiking neural networks. *bioRxiv preprint* 2026.02.12.705507. | https://doi.org/10.64898/2026.02.12.705507 | verified | **The predictive-coding variant of e-prop**: the "third factor" is replaced by the **local prediction error of predictive coding itself**. Self-description, verbatim: "We term the resulting model 'Predictive E-prop', emphasizing its role as **a learning principle rather than a task specific model**." On three tasks over two dynamical systems (a sinusoidal limit cycle and Lorenz) it matches truncated BPTT (p > 0.05) while converging in **70% fewer epochs** (about 23 vs about 80); performance stays stable up to σ_in ≤ 0.2. ⚠️ **Zero occurrences** of metacognition / forgetting / working memory in the paper -- v6.2 of the project plan listed it under phase 4's "metacognitive gating", which was a misattribution; v6.3 corrects this. ⚠️ Licence `cc_no` (All rights reserved), so **its figures may not be reused**. ⚠️ Near-name collisions: the same group published another e-prop paper in 2025 (*Neuromorphic Computing and Engineering* 5(4) 044002, on separating connectivity from intrinsic noise) -- do not swap their conclusions; Ororbia's spiking neural predictive coding is yet another line |
| 20 | Graf, L., Su, Z., & Indiveri, G. (2024). EchoSpike Predictive Plasticity: An Online Local Learning Rule for Spiking Neural Networks. *arXiv preprint* arXiv:2405.13976. | https://arxiv.org/abs/2405.13976 | verified | **ESPP = EchoSpike Predictive Plasticity**: the whole previous sample's spiking activity serves as the prediction target (the "echo"); same-label samples are pulled together and different-label ones pushed apart. A predictive-plus-contrastive **inter-layer local rule** that never uses automatic differentiation. 84.32% on SHD (self-reported). **The original source of the Chinese phrase "event priority" is its §III-C**: input-activity and loss thresholds let the rule itself decide which time steps get a weight update -- verbatim "ESPP intrinsically has the ability to selectively choose those time steps that matter the most" -- measured at only **18%-27%** of time steps, decreasing with training. ⚠️ It is **another rule**, not an e-prop variant (the same treatment ADR-0010 gave TP); do not put it in the §3.1 e-prop chain. ⚠️ The GitHub link in the paper's footnote (`largraf/EchoSpike`) is **404**; the authoritative repository is [Zhe-Su/ESPP](https://github.com/Zhe-Su/ESPP) (**Apache-2.0**, though its LICENSE still carries the unfilled copyright placeholder). ⚠️ "ESPP" is a high-frequency acronym (employee stock purchase plan, European Strategy for Particle Physics, European Sustainable Phosphorus Platform, Espressif's component library) -- in an SNN context it is this one |
| 21 | Frenkel, C. (2022). eprop-PyTorch: PyTorch implementation of the eligibility propagation (e-prop) learning algorithm. *GitHub Repository*. | https://github.com/ChFrenkel/eprop-PyTorch | verified | The "e-prop implementation reference" named by the plan's §1.4/§8. **Apache-2.0** (copyright University of Zurich; the LICENSE is the untouched upstream template, its Appendix still reading `Copyright [yyyy] [name of copyright owner]`). **Not official** -- the official one is [15]; **not on PyPI and with no tags/releases**, so it can only be anchored at commit `0f32a8f2` (a single commit dated 2022-02-18; 7 files in total). **LIF only, with ALIF explicitly removed** (`main.py`: "Support for the ALIF neuron model has been removed."), and the only task is evidence accumulation; **no dependency manifest, no CI, no tests**, and two uses of `np.int` in `setup.py` now fail on NumPy ≥ 1.24. ⚠️ Therefore **ALIF must be checked against [15] and the original [1]** -- the sentence in ADR-0008 decision 3 that proposed using this repository as the ALIF reference was withdrawn in that ADR's later correction note |

## To-do

### ✅ P0: TD-LTP provenance verification — done, and the conclusion is "a provenance exists"

§3.2 and §11 of the project plan write the Critic's training rule as "TD-LTP" and mark it
"⚠️ provenance to be added", and §11 stipulates that **"if there is no provenance, phase 5
does not get approved as a project"**. The verification conclusion:

**TD-LTP is a learning rule formally named by its authors, its provenance is exact, and the
P0 gate is passed.**

- **Provenance of the naming**: Frémaux, Sprekeler & Gerstner (2013), *PLoS Comput Biol*
  **9(4)**:e1003024 — that is, the newly added [12] in this table. The original contains an
  explicit naming statement, the Eq. 17 number and the Figure 2A caption; this is not a loose
  term dressed up as a formal name. The paper is open access, and "TD-LTP" appears 40 times
  in the full text, so you can re-verify it yourself.
- **The project plan's error is a mispaired citation, not fabrication**: the project plan binds
  "TD-LTP" to **[3] Frémaux et al. 2010**, but the name comes from the **2013** paper. The two
  papers do different jobs and should be cited separately:
  - **The name TD-LTP and the rule itself** → [12] Frémaux et al. 2013;
  - **R-STDP has an unsupervised bias, and stimulus-specific reward prediction must be
    introduced** → [3] Frémaux et al. 2010 (which is exactly the subject of Figure 3 in that
    paper).
- **A precedent for its use**: [13] Tihomirov et al. 2025 uses TD-LTP as the Critic of a
  spiking actor-critic, which is isomorphic to the architectural choice of §3.2 in the project
  plan, and can serve as external evidence that this route is viable.

**A disagreement that arose during verification (recorded truthfully)**: two of three
independent search angles attributed TD-LTP to Tihomirov et al. 2025, conflicting with [12].
An adversarial re-verification overturned both of those attributions — the reviewer admitted
that they "found no usage of TD-LTP before Tihomirov/Rybka", i.e. their search did not go back as
far as 2013. The attribution in [12], by contrast, was **reproduced verbatim** by the reviewer after
independently fetching the PLoS full text. So [12] is accepted. This also shows that
**one search failing to find something does not mean it does not exist**, and that
attribution-type conclusions must go through an adversarial re-verification.

**Recording discipline that the review also pointed out** (putting it into the ADR avoids
overclaiming):

1. The sentence "no back-propagation signal has been observed in experiments" in Frémaux 2013
   refers to the credit assignment property of the TD error along **time**; it is **not** a
   statement that "biological networks do not back-propagate an error vector". Using it to
   argue for "purely local" takes the sentence out of context; cite the three-factor form
   itself instead (pre×post → κ filtering → multiply by the scalar δ) together with the
   original's phrase "the global signal".
2. For the relationship between TD-STDP and TD-LTP the original's wording is "behaves
   similarly" / "only slightly worse"; it should **not** be phrased as "functionally
   equivalent".

### Candidate implementation baselines for phase 1 (not yet introduced into the project plan, hence unnumbered)

The following works were confirmed during verification to exist and to be described
accurately, but are not yet cited by the project plan, so they are not numbered for now:

- Chung & Kozma (2020), *Reinforcement Learning with Feedback-modulated TD-STDP*,
  arXiv:2008.13044 — open access, with CartPole-v1 and LunarLander numbers, and an ablation
  showing that it does not learn at all once feedback modulation is removed. **The cheapest
  starting point to reproduce locally.**
  Note that its critic is a **population of neurons** rather than a single neuron (those
  numbers come from Vlasov et al. 2024).
- The e-prop RL part of Bellec et al. (2020) (this table's [1]) already contains an
  actor-critic and can replace "R-STDP + a separate Critic" wholesale; it is another route
  worth evaluating.

### Verification procedure

For each citation, verify four things:

1. **Existence** — the URL / DOI / arXiv number is reachable;
2. **Metadata** — author, year, journal or conference, volume-issue-pages/article number match
   what is registered;
3. **Attribution** — no conclusion of A has been recorded against B (the lesson of [8] and
   [10]/[11]);
4. **Numbers** — the specific numbers attributed to it in the body of the project plan can be
   found in the original. If they cannot be found or do not match, the status can only be
   `partial`, and the notes must state which item did not match.

**Do not substitute memory for search.** If you cannot verify something, mark it honestly as
`unverified` or `partial`.

## Items in the body of the project plan awaiting correction

The following problems belong to the **body of `BioSNN-Plug_项目计划书_v6.2.md`**, which this file cannot change. **2026-09-23: all of these corrections are now folded into [`BioSNN-Plug_项目计划书_v6.3.md`](../BioSNN-Plug_项目计划书_v6.3.md)** — the table is kept as a change log, and **v6.2 remains the version to cite as the plan; use v6.3 when you need the corrected wording**.

| Location | Problem | Suggested change |
| :--- | :--- | :--- |
| §3.2, the §6 table, §10, §11, §12 | "TD-LTP Critic" bound to [Frémaux et al., 2010] | Split into two: TD-LTP → 2013 PLoS Comput Biol 9(4):e1003024; the R-STDP bias → 2010 J Neurosci. Also **delete the "provenance to be added" mark and the P0 to-do of §11** |
| §3.1 key correction 1 | "on MNIST and SHD" | Change to **N-MNIST** and SHD; add the journal version's provenance |
| §5.2, §7 phase 3 | The provenance in the row containing "PS-SNN … 76.42%" | Article number 42970 → **12653** |
| §6.1, §7 phase 6 | Hajizada et al. "2026" + 70×/5,600× | Change the year to **2025**; align the figures with the version cited (v2 gives 113×/6,600×) |
| §3.3 | "does not suffer a deep representation bottleneck" | Rewrite to follow the original as "avoids the representation-bottleneck problem of deep networks" |
| §2.2 | "dual-channel fusion" attributed to TAAF | What TAAF provides is temporal-attention-guided adaptive fusion; **dual-channel fusion is this project's own design** and should not be attributed |
| §4.6, reference [11] | ASTRA's author written as the project name | Change to the GitHub account christophejlegros-lgtm |
| §4.3, reference [10] | MEMBRAIN's author written as the project name | Change to the GitHub account tfatykhov |
| §3.2 | The concrete form of the Critic's training rule | Write it according to the original of [12] (2013): Δw ∝ δ(t)·κ∗[x_i·y_j] (**counting pre-before-post only**), where δ is the global scalar TD error. And avoid using the "no back-propagation signal" sentence to argue for locality |
| §3, the §8 selection table, the §11 risk table, §3.1 key correction 1 | Trace Propagation is written as "e-prop's storage optimisation: O(N²) → O(N)" | Rewrite per [ADR-0010](adr/ADR-0010-eprop-quadratic-storage-and-trace-propagation.en.md): TP is **another rule** (traces per neuron, not per synapse), not a memory-saving version of e-prop; the two are parallel, not substitutes. Also drop §3.1's framing of it as this item's remedy |
| §12.1, §1.4, §8 | The license table writes SpikingJelly as "same license as this project (Apache-2.0)" | Rewrite to the facts: SpikingJelly uses the **Open-Intelligence Open Source License 1.0** (OIOSL); add its commercial-use disclosure obligation. In the technology-choice rows of §1.4/§8, note "the license terms of this dependency, and their effect on the development environment's Python floor, are in `docs/adr/ADR-0008`" |
| §7 phase 4 task line | **Predictive E-prop** listed alongside "metacognitive gating" | That paper contains **zero** occurrences of metacognition / forgetting / working memory, so the pairing is a misattribution. v6.3 keeps metacognitive gating (grounded in this project's own §2.2/§3.4) and moves Predictive E-prop, per its self-description ("a learning principle"), into the §3.1 list of e-prop variants; the task line now carries **event priority** (ESPP's selective time-step updates) |
| §3.1, §8 | "Event priority" written as if it were a term from some paper | No such term exists in the 2025-2026 spiking literature (seven papers checked word by word; `priorit*` hits 0/0/0/1/0/0/0). v6.3 writes it as ESPP's mechanism and states that "event priority" is this project's Chinese paraphrase, not a literature term |

# 02 · Skeptic review: three hostile-but-fair ICML 2027 reviewers

*Phase 2 of the novelty/methodology upgrade · written 2026-10-22 · Skeptic agent · read-only pass (no code run, no
repo file edited except this one).*

**Inputs read in full:** `PLAN_V7.md`, `improvement.md`, `research/spikes/RESEARCH.md`, and the Bai, Liu & Li, Marin
and OPEN-1B JSONs in `spike-related-work/results/`. I also pulled the novelty-threat and scale fields from all 16
rival JSONs, and read `research/experiments/proxy/config.yaml`, `research/baselines/reset.py`, and parts of
`spikes/natural.py` and `spikes/capture.py`. **External:** ICML 2026 reviewer instructions + CFP, OpenReview API
review text for 7 comparable papers, the Marin 32B retro, the PaLM paper, and Wortsman et al. 2024.

**Evidence labels.** **[V]** = verified this session against the source (file or web page).
**[V-prior]** = verified by the 2026-10-19 deep pass (`spike-related-work/`), not re-read today.
**[I]** = inferred (my arithmetic or reasoning; it can be wrong). **[A]** = assumed. **UNVERIFIED** = not confirmed.

> **Scope note.** `improvement.md` already lists the kill-test holes (A1 `Bs≡B0`, A2 no held-out loss, A3 no
> thresholds, A4 no natural `B*`), the ROCm checklist (A6), the HVP/SDPA problem (A7) and the 2026 prior-art map
> (§2.3). This review **does not repeat them**. It goes after what a reviewer would hit *after* those are fixed.

---

## 0 · Bottom line

1. **The regime is the biggest threat, not novelty.** The "11M proxy" is about **1.33M non-embedding parameters in
   3 layers** [I, from config]. Its 600-step trunk sees about 0.2–0.9 tokens per parameter, at constant LR with no
   warmup or decay [V]. Reviewers of directly comparable papers rejected or down-scored models trained to *7.7*
   tokens/param as "really undertrained" (§2).
2. **The thesis ("damage persists in m/v after the weights recover") is never directly tested by the planned
   primary endpoint.** At β2 = 0.95, a spike's contribution to `v` decays to 0.6% within 100 steps [I]. Any damage
   that lasts longer must live in the weights or trajectory, unless the state-patching decomposition (RESEARCH §4.2,
   designed but not in the V7 battery) shows otherwise.
3. **"Poisoned direction" is not well defined.** Bai et al.'s mechanism says pre-spike `v` is too *small* (the
   cause). The post-spike inflated `v` is the system's own stabiliser. Our `v ← v̄` and v-vs-v̄ scoring may be
   "repairing the cure". `v` is diagonal, so "rank-limited directions of v" is not representable as stated [I].
4. **"Beats full reset" is a strawman bar as built.** `Bg` keeps Adam's step counter [V]. The repo's own notes
   predict mis-scaled post-reset updates for it [V]. The production baseline is rollback-and-skip (PaLM) [V], and
   it is missing.
5. **Projected ICML outcome if executed exactly as planned: overall 3/6 (weak reject).** With the five must-fixes in
   §3, it is plausibly a 4 (weak accept, borderline). This matches V7 core value 6's own TMLR-shaped ceiling.

---

## 1 · What ICML actually asks reviewers to judge

ICML 2027's call is **not yet published** [V: 404 on icml.cc/Conferences/2027/CallForPapers, 2026-10-22]. An
unofficial tracker estimates **abstract 2027-01-16, paper 2027-01-22 AoE** ("Not yet official", last reviewed
2026-10-08) [V for what the tracker says; the date is UNVERIFIED]. The 2026 criteria, assumed to carry over [A]:

1. **Four sub-scores, 1–4:** **Soundness** ("Are claims well supported (e.g., by theoretical analysis or experimental
   results)?"; "Soundness is distinct from impact"), **Presentation** (clarity + positioning vs prior work),
   **Significance** ("advance[s] understanding, capabilities, or practice in machine learning"), and **Originality**
   ("does not necessarily require introducing an entirely new method"; novel insights from evaluating existing
   methods count) [V].
2. **Overall, 1–6:** 4 = "Technically solid paper that advances at least one sub-area of AI". 2 = reject for
   "technical flaws, weak evaluation, inadequate reproducibility" [V].
3. **Experiments:** reviewers check whether "hypotheses are actually tested by the experiments" and whether
   "empirical claims do indeed follow from empirical results". There is **no numeric scale or compute bar** [V].
   CFP: "All claims must be clearly stated and supported by reproducible experiments and/or sound theoretical
   analysis" [V].
4. **Prior and concurrent work:** works public **< 2 months before the full-paper deadline are concurrent** (not
   required). Everything older is fair game [V]. With a 2027-01-22 deadline, that cut-off is about **2026-11-22**
   [I]. So Liu & Li (2026-08-20), OPEN-1B (2026-09-15), the Megatron determinism blog (Oct 2026) and Ramasinghe
   (Jun 2026) are all **prior work we must discuss** [I]. Limitations: "Authors should be rewarded rather than
   punished for being up front" [V].
5. **Format:** 8 pages + unlimited references and appendix, and a required impact statement [V, 2026 CFP]. The
   ICML 2025 form had explicit fields `claims_and_evidence`, `experimental_designs_or_analyses`,
   `essential_references_not_discussed` [V, field names from the OpenReview API]. Expect reviewers to think in those
   terms even under the 2026 four-axis form.

Links: https://icml.cc/Conferences/2026/ReviewerInstructions ·
https://icml.cc/Conferences/2026/CallForPapers · https://www.opencurious.com/ai-conference-deadlines/icml-2027
(all fetched 2026-10-22).

---

## 2 · What real reviewers wrote on comparable optimizer-instability papers

Quotes come from the OpenReview search API (`api2.openreview.net/notes/search`, fetched 2026-10-22). The fetch tool
capped each quote at about 120 characters, so these are **verbatim fragments as returned, not checked against the
rendered forum pages**: forum pages sit behind a bot challenge. I could not reach ICML 2026 reviews (Bai
`STWQoscanw`, AdaGC `uBYlCu8b8Z`) through search; they may not be public. **UNVERIFIED.**

| Paper (venue, outcome) | Reviewer · score | Quote | Maps to objection |
|---|---|---|---|
| **Bai et al.**, *Adaptive Preconditioners Trigger Loss Spikes in Adam* — ICLR 2026 **rejected**, then ICML 2026 accepted [V] · [forum](https://openreview.net/forum?id=dwx3rjWGju) | ewU8 · 6 | "novelty w.r.t prior works, notably https://arxiv.org/abs/2207.14484 (which was cited)" | #6 novelty |
| | xauY · 2 | "The fact that $\beta_2$ influences the spikes was also shown in prior work [3]" · "Missing experiments on real data. Only one of the experiments is done on real data." | #6, #9 |
| | kJgM · 2 | "is computationally prohibitive and thus unrealistic for modern deep learning." (Hessian monitoring) | #10 cost |
| | wQk8 · 8 | "The paper lack experiments comparing Adam with other optimizers or its common variants" · "The large-scale experiments rely on synthetic data or controlled training conditions." | #13, #9 |
| **SPAM** — ICLR 2025 poster [V] · [forum](https://openreview.net/forum?id=L9eBxTCpQG) | BUJ8 · 6 | "There is no comparison between Spike-Aware Clipping and simply nullifying gradient spikes." · "Were the fine-tuning experiments conducted using multiple random seeds?" | #5 bar, #7 stats |
| | BU76 · 6 | "Clipping gradients based on a threshold seems to lack novelty." | #6 |
| | LnTM · 6 | "The choice of the gradient spike threshold might affect performance to a great extent." | #3 thresholds |
| **AdaGC** — ICLR 2026 **rejected**, then ICML 2026 accepted [V] · [forum](https://openreview.net/forum?id=ZQcDUhEOg9) | 9419 · 4 | "Since the experiments do not demonstrate that spikes occur" · "SPAM is compared only on downstream tasks but not in spike-score evaluation" | #9, #5 |
| | J259 · 4 | "The total token counts (e.g., 36B tokens for LLaMA-2 7B) are small compared to real large-scale pretraining" · "treats gradient spikes as a black-box phenomenon, lacking diagnostic analysis" · "gains of AdaGC when applied to the Lion and Muon optimizers are minimal" | #1, #13 |
| | G9Tz · 4 | "Comparison with stronger and more modern baselines" | #5 |
| **Stable-SPAM** — ICLR 2026 **withdrawn** [V] · [forum](https://openreview.net/forum?id=BuJ2C6iZRG) | zR6K · 4 | "The 1B models are also really undertrained, only training to 7.7 TPP" | #1 |
| | YxM2 · 4 | "Experiments are confined to small-scale settings" · "The paper provides insufficient analysis of the root causes behind 4-bit training instability." | #1, #2 |
| | vRDd · 2 | "Were multiple runs with different seeds performed for smaller models?" · "At least one long-token, large-scale run (chinchilla optimal) should be provided" | #7, #1 |
| | PV1f · 4 | "what about recent Muon optimizer" | #13 |
| **Sharpness Disparity Principle** — ICML 2025 poster [V] · [forum](https://openreview.net/forum?id=DZ6iFdVDrx) | umnV · 3 | "It remains unclear whether this strategy generalizes to massive-scale models." · "Most experiments stop at 100K steps." | #1, #11 |
| | jDEo · 4 | "the authors only report the loss and do not evaluate the downstream performance" | #11 |
| **Understanding Sharpness Dynamics … Minimalist Example** — ICML 2025 poster [V] · [forum](https://openreview.net/forum?id=XfjrLEPOQV) | 2e3Z · 4 | "This limits how representative the results are." | #1 |
| **Small-scale proxies** (Wortsman) — ICLR 2024 oral [V] · [forum](https://openreview.net/forum?id=d8w0pmvXbZ) | hm5n · 8 | "it doesn't demonstrate any new insights regarding large models." | #1 |

**Pattern [I].** These reviewers repeat four objections:

- the scale or token count is too small;
- the method treats spikes as a black box, with no root cause;
- a stronger or simpler baseline is missing;
- there are not enough seeds.

The two papers closest to ours (Bai, AdaGC) were both *rejected at ICLR 2026 and accepted at ICML 2026* on
resubmission. That shows the critiques are survivable when they're answered. It also shows a first submission
usually draws them.

---

## 3 · The 15 objections, ranked

**Personas:** **R1** = methods/soundness, **R2** = novelty/positioning, **R3** = significance/scale.

**Cost conventions [I].** GPU cost is given in **P100-equivalent hours** from V7's measured 0.0619 s per proxy step
(batch 16). 1 P100-h ≈ 58k proxy steps. MI300X time should be ≤ this if MI300X is at least as fast as a P100 on
these shapes [A]. "Fork" means 300 steps unless stated otherwise. Person-hours are rough.

**MUST** = must be addressed for an ICML accept (five of them).

### FATAL

#### F1 · The proxy regime is too far from LLM pretraining to support the headline claim — R3 · **MUST**
- **Objection.** Per `config.yaml` [V]: `n_layer 3, n_embd 192`, vocab 50,304, tied embedding. That gives about
  11.0M parameters, of which **≈ 9.7M (≈ 88%) are the token embedding** and ≈ 1.33M are transformer blocks [I,
  arithmetic]. The trunk is 600 steps (`capture.SITES = (160, 480)` "600-step trunk" [V]). At batch 16–64 × 256
  tokens, that is **2.5–9.8M tokens ≈ 0.2–0.9 tokens/param**, against about 20 for Chinchilla [I]. The LR is
  **constant 3e-4 with no warmup or decay** (`natural.py:29`: "the proxy config has no warmup/decay" [V]). Spike
  sites 160 and 480 sit in what is still the very early phase of training. The 124M "smoke" at ~64–68 P100-s per
  optimizer step [V, V7 §3] gives n ≈ 1–3 instances at best. Bai et al. place the decoupling mechanism "late in
  training" [V-prior], and our trunk never reaches late training.
- **Reviewer precedent:** Stable-SPAM ("only training to 7.7 TPP"), AdaGC ("token counts … small"), Sharpness
  Disparity ("generalizes to massive-scale models?"). See §2.
- **What answers it:**
  - (a) Re-define the proxy trunk with warmup + cosine to a fixed end, at ≥ ~20 tokens per *non-embedding*
    parameter (≈ 27M tokens ≈ 6.5k steps at batch 16).
  - (b) Add one middle rung (e.g. 6 layers, d = 384: ≈ 10.6M non-embedding, ≈ 30M total [I]) with key branches only.
  - (c) 124M on the primary contrast only, n ≥ 3, after measuring MI300X speed.
  - (d) Score C2 offline (no fork) on **OPEN-1B**'s full-optimizer-state checkpoints bracketing its natural
    grad-norm events (ckpt 1700 → event at step 1782; ckpt 1800 would be post-event [I]). Caveats [V-prior]: the
    events are "modest grad-norm excursions", whether they were loss spikes is unverified, and replay needs
    proprietary RepOps + int8 QAT. Whether the bucket really holds m/v is UNVERIFIED.
- **Cost [I]:**
  - (a) ≈ 0.6 P100-h for trunks + ≈ 2.5 P100-h to re-run the battery;
  - (b) ≈ 4–6 P100-h (step time assumed ~0.15 s [A]);
  - (c) ≈ 20+ P100-h, so not inside the 15 h ask;
  - (d) ≤ 1 GPU-h + ~19 GB download + 10–15 person-h.
  - Total 15–25 person-h.
- **Before deadline?** (a), (b), (d): **yes**. (c) at n ≥ 3: **only with the ~85 h follow-on ask landing by early
  December** [I].

#### F2 · The core thesis is never directly tested, and state poisoning self-heals within about β2's memory — R1 + R3 · **MUST**
- **Objection.** The claim is that damage "may persist after the weights recover" (V7 §2). With β2 = 0.95, the
  spike's share of `v` falls as 0.95ᵏ: **36% at 20 steps, 7.7% at 50, 0.6% at 100** [I]. For a κ-fold inflation,
  `v` returns to within 2× of normal after about ln κ / 0.0513 steps: **≈ 90 steps for κ = 100, ≈ 180 for
  κ = 10⁴** [I]. `m` (β1 = 0.9) forgets in about 10–50 steps.
  - So any Δ that lasts beyond about 200 steps cannot be carried by m/v. It is weight or trajectory damage, which
    a selective m/v repair can only reach indirectly.
  - Marin's 32B diagnosis points the other way from our story: "update norm spikes always preceded grad norm
    spikes" [V], with m doubling and v flat [V-prior].
  - The battery (V7 §4.3) compares repairs. It never runs the decomposition that tests the thesis:
    w-only vs (m,v)-only patching, RESEARCH §4.2, which is "design" only.
  - Site 480 of a 600-step trunk leaves only **120 steps** of horizon if forks stop at the trunk end. That is
    shorter than the paper's own k½ ≈ 158 prediction [I; fork horizon not pre-registered, improvement A3].
- **What answers it:** Make **optimizer-state patching** a pre-registered primary experiment. At t = spike_end +
  {10, 50, 150}:
  - transplant (m,v) from `B*` into `B0`, keeping `B0`'s w ("denoising");
  - transplant w only;
  - also do the reverse, `B0`'s state into `B*` ("noising"; the standard activation-patching pair);
  - report the fraction of Δ recovered, with paired CIs, at the horizons in M11.

  Add an off-manifold control, because (m,v) from one trajectory paired with w from another is an inconsistent
  state: patch `B*`'s (m,v) into `B*` taken at a ±5-step offset, and show the patch itself does no harm.
- **Cost [I]:** 30 instances × 6 patch types × 2 times ≈ 360 forks ≈ 1.9 P100-h, plus a greedy oracle over ~20
  tensor groups on 10 instances ≈ 1 P100-h. 10–15 person-h; no harness change (RESEARCH §4.2).
- **Before deadline?** **Yes.** It is the cheapest high-value item in this review.

#### F3 · "Poisoned direction" is not well defined: wrong sign, wrong basis, circular — R1 · **MUST**
- **Objection, three parts [I]:**
  1. **Sign.** In Bai et al. the *cause* is `v` decaying **below** g² (λ_max(Ĥ) > 2/η). The large post-spike
     gradients then *refill* `v` and end the spike (their stage 4) [V-prior]. The post-spike inflated `v` is the
     stabiliser. C2 scores `v` against the pre-spike `v̄`, and `Bvbar` sets `v ← v̄` (`reset.py` [V]). That moves
     `v` back toward the pre-spike, under-preconditioned state that Bai et al. say *triggers* spikes, so it risks
     re-inducing one. RESEARCH §5's minimal stabilising repair scales `v` **up** (a ≥ 1), which is the opposite
     sign from "v is poisoned upward". The plan holds both stories at once.
  2. **Basis.** `v` is an elementwise (diagonal) preconditioner. The top-k eigenvectors of
     `H~ = D^-1/2 H D^-1/2` are dense, and a "rank-k repair of v along u_i" gives a non-diagonal preconditioner
     that AdamW cannot store. For `m`, subspace projection is well defined. For `v`, only **support-limited**
     (coordinate, row, head or tensor) edits are representable.
  3. **Circularity.** `D` is built from the poisoned `v̂`, so the eigenbasis used to find poison is defined by the
     poison.
- **What answers it:**
  - Split "poison" into two signed, operational failure modes:
    - **under-preconditioned:** `v_i` ≪ the g² it should track, high λ_grad(Ĥ). This is the cause.
    - **over-preconditioned:** `v_i` ≫ `v̄_i`, update stall. This is the consequence.
  - Define the ground truth *causally* as the components whose patching (F2) recovers Δ. C2 is then an
    unsupervised predictor of that oracle, scored by precision/recall/AUROC.
  - For `v`, say "support-limited", not "rank-limited".
  - Compute `H~` with both `v̂_post` and `v̄` and report the agreement.
  - Classify every instance by failure mode before repairing.
- **Cost:** ≈ 0.1 P100-h for λ_grad probes; ~8–12 person-h of definitions and math.
- **Before deadline?** **Yes**, but it must happen **before C2 code is written**: it changes the output contract
  (CLAUDE.md §7.2).

### MAJOR

#### M4 · QK-Norm / attention-stack confound (Marin 32B, Wortsman 2024) — R3 · **MUST**
- **Objection.** Marin: "the instability was rooted in the model's attention stack rather than in optimizer state",
  "the loss spikes disappeared entirely" after QK-Norm, and clipping/skip-step "softened the spikes but never removed
  them" [V, retro page]. Wortsman et al. (ICLR 2024): attention-logit growth instabilities "also appear in small
  models when training at high learning rates", and "mitigations previously employed at large scales are equally
  effective in this regime" [V, abstract]. Our nanoGPT has plain LayerNorm and no QK-Norm [V,
  `model/nanogpt.py`]. The natural stress presets run at 5–10× LR [V], which is exactly Wortsman's regime. So proxy
  spikes may be attention-logit blow-ups, with m/v damage as a downstream symptom.
- **What answers it:**
  - A QK-Norm proxy variant (a model change, not a harness change; still flag it).
  - Re-run the natural stress search and an lr_bump/corrupt_batch subset on it.
  - Report the spike rate with and without QK-Norm, and the F2 state-carried damage fraction with and without.
  - Log the max attention logit as telemetry (free).
  - State the claim as "damage carried in optimizer state", not "root cause" (already suggested in improvement
    §2.3; this makes it an experiment).
- **Cost [I]:** ≈ 2.1 P100-h (120k-step stress search) + ≈ 0.3 P100-h (subset battery) + determinism gate re-run.
  6–10 person-h.
- **Before deadline?** **Yes.**

#### M5 · "Beats a full reset" is a strawman; the real bar is rollback-and-skip and the best cheap fix — R1 + R3 · **MUST**
- **Objection.**
  - `zero_moments` keeps Adam's per-parameter `step` [V, `reset.py`]. Bias correction therefore no longer matches
    the zeroed moments. The docstring itself predicts first updates "~1/sqrt(1−beta2) larger … (~4.5× at
    beta2=0.95)" for the v-only reset [V], and RESEARCH §4.1 predicts that `Bg` "pays a global ~4.5× update
    overshoot" [V]. Beating a baseline that is mis-specified by construction tells a reviewer nothing.
  - Production practice is rollback-and-skip. PaLM "re-started training from a checkpoint roughly 100 steps before
    the spike started" and "skipped roughly 200–500 data batches" [V]. OPEN-1B's abandoned first run used a full
    optimizer reset at resume [V-prior].
  - C1 already keeps snapshots, so a reviewer will ask why we don't simply roll back.
- **What answers it:**
  - Add three branches:
    - **rollback-and-skip**, from the ring-buffer snapshot k steps before the detector t0, skipping M batches.
      This intentionally shifts data, so label it as the practitioner baseline, not `B*`.
    - **Adam-MR** (m + v + t reset).
    - a **bias-correct `Bg`**.
  - Pre-register the primary contrast as **repair vs the best non-oracle baseline**, picked on seed 0, and report
    the gap closed, G = (Δ_B0 − Δ_branch)/Δ_B0, with cost in recomputed steps.
- **Cost [I]:** ≈ 0.5 P100-h; 4–8 person-h.
- **Before deadline?** **Yes**, but it must be pre-registered before battery data (V7 core value 3).

#### M6 · Novelty compression: every ingredient has 2026 prior art — R2
- **Objection [V-prior]:**
  - **Bai et al.** own the mechanism and the λ_grad(Ĥ) > 2/η predictor, which is the curvature half of C2's
    score, and they show block-level v-decay before spikes.
  - **Liu & Li** run paired control/shock trajectories with a shared future and inject into {θ, m, v} channels,
    which is C1's attribution design.
  - **Megatron Core** and **OPEN-1B** show bitwise replay at 2,432 GPUs and 1.6B.
  - **SPAM** resets moments.
  - **Spectra** reports that ~97% of the spectral energy sits in a top-1.5% subspace, so "poison is low-rank" is
    half-anticipated.
  - What remains is *localization inside m/v + selective repair + causal verification*. If C3 only ties strong
    baselines, the residual contribution is "a careful measurement study".
  - The ICLR 2026 reviewer ewU8 hit Bai on exactly this kind of overlap (with Cohen et al., 2207.14484) [V].
- **What answers it:**
  - Run Bai's predictor and **Bai-style per-block v-decay as the baseline localizer**. C2 must beat it on the F2
    oracle.
  - Add Spectra's energy-only damping as a control.
  - Optionally use Liu & Li's first-order ISO as a comparator, to show where linearisation breaks at spike scale.
  - Claim C1 as a *counterfactual state-editing instrument*, not as determinism (already in improvement §0.4).
  - Lean on ICML's originality clause: insight from evaluating methods counts [V].
- **Cost:** ≈ 0.5 P100-h; 15–25 person-h (largely writing).
- **Before deadline?** **Yes.**

#### M7 · n = 10 per type is really 2 data contexts × 5 inits, and the multiplicity is unbudgeted — R1
- **Objection [I, from V7 §4.2]:**
  - Seeds change "the initialization only"; data order is identical [V]. The 2 sites share each seed's trunk, so
    instances are nested in seeds (pseudo-replication), with 5 clusters, not 10 independent units.
  - Seed 0 tunes the detector thresholds, so detector-timed branches (`Bskipstep_det`) have only 8 held-out
    instances per type.
  - Minimum detectable paired effect at 80% power, two-sided α = 0.05:
    - d_z ≈ 1.0 at n = 10 and ≈ 1.15 at n = 8;
    - ≈ 1.5 with Bonferroni across ~13 branches [I; normal-approximation arithmetic, not run in software].
  - With a 0.01-nat effect floor (V7 §4.4), that needs an SD of paired differences ≤ ~0.007–0.01 nats. Nobody has
    measured that.
  - Reviewers asked SPAM and Stable-SPAM about seeds (§2).
- **What answers it:**
  - **One** pre-registered primary contrast (M5), with Holm correction for the secondaries.
  - A mixed model: Δ ~ branch + site + (1 | seed).
  - Add a **data-order seed**, a permutation key on `get_batch`: B* alignment still holds inside a seed.
  - Raise to 10 seeds for the primary contrast only.
  - Run a pilot to estimate SD before locking the n.
- **Cost [I]:** ≈ 0.7 P100-h; 6–10 person-h.
- **Before deadline?** **Yes.**

#### M8 · A bitwise Δ is one sample from a chaotic map; there is no placebo distribution — R1
- **Objection [I].** Near a spike, training is at or past the edge of stability, where small perturbations grow.
  Determinism removes run-to-run noise, but any edit (relevant or not) can produce a large |Δ| through divergence
  alone. A Δ from one replay is then a single realisation, not an expected causal effect. `Br` (random subspace)
  helps only if it is matched in edit norm *and* support size.
- **What answers it:**
  - For K = 5 nuisance replicates per branch, add a 1-ulp / 1e-7-relative perturbation to w at fork.
  - Report the within-instance spread for B0, B*, repair and Br.
  - Require the repair effect to exceed the matched-placebo distribution.
  - This also turns the "bitwise" instrument into a strength: it lets us *measure* sensitivity, not just remove it.
- **Cost [I]:** ≈ 1.0 P100-h (10 instances); 3–5 person-h.
- **Before deadline?** **Yes.**

#### M9 · Induced spikes may be the wrong failure mode, or trivially easy to localize — R3
- **Objection [I]:**
  - `lr_bump` (10× LR for a window [V]) produces large gradients that inflate `v`. After the bump the state is
    **over**-preconditioned, the opposite of Bai's natural cause (F3).
  - `corrupt_batch` (uniform random tokens) mostly hits rare-token **embedding rows**, which have sparse gradients.
    Finding them is trivial and says little about a general localizer.
  - `v_inflate` plants the answer in `v`. That is fine as a C2 unit test, but circular as evidence that spikes
    poison `v`.
  - The stress presets (5–10× LR, β2 = 0.999, no clip [V]) are hyperparameters nobody trains with. RESEARCH §7
    already concedes they are not "production" spikes.
  - Reviewers: AdaGC 9419, "do not demonstrate that spikes occur"; Bai xauY, "Missing experiments on real data" (§2).
- **What answers it:**
  - Report every result per type; never pool.
  - Classify each instance's failure mode (F3) and show the induced and natural mechanisms match before
    generalising.
  - Make `repeat_ngram` (OLMo-documented cause) a primary type if it qualifies.
  - Add the external checks (OPEN-1B offline, K2-V2 logs).
- **Cost:** ≈ 0.3 P100-h; 6–10 person-h.
- **Before deadline?** **Yes.**

#### M10 · Localization costs more than just rolling back — R3
- **Objection.**
  - HVPs need the SDPA math backend (improvement A7).
  - Gradient-directional SNR needs several gradient samples.
  - A Lanczos top-k with k ≈ 20 needs ≈ 60+ HVPs at ≈ 2–3 step-equivalents each, plus ~32 gradient samples. That
    is roughly **150–250 step-equivalents per localization** [I], against **≈ 100 recomputed steps** for PaLM-style
    rollback [V for PaLM's figure].
  - Bai et al. themselves say HVP analysis above ~200M is "computationally demanding" [V-prior]. ICLR reviewer kJgM
    called Hessian monitoring "computationally prohibitive" [V].
- **What answers it:**
  - Report a **recovery-per-GPU-second Pareto**.
  - Ablate the localizer (no-HVP `v/v̄` score · SNR only · curvature only · full). If the cheap score matches, say
    so: honesty beats a hidden negative.
  - Use a warm-started power iteration (Ramasinghe: ~5 HVPs/step [V-prior]).
  - Time it once at 124M.
- **Cost [I]:** ≈ 0.5–1 GPU-h; 6–8 person-h.
- **Before deadline?** **Yes.**

#### M11 · The horizon decides the result; Δ at a short horizon isn't significance — R3
- **Objection (builds on improvement A2).** Even with held-out loss:
  - A Δ measured inside the state-memory window (F2) can vanish by the end of training. With constant LR there is
    no decay phase to show whether it does.
  - The 0.01-nat floor at an unreported horizon is not a practitioner outcome.
  - ICML 2025 reviewers flagged "only report the loss" and runs that "stop at 100K steps" (§2). At 11M, downstream
    tasks are meaningless, so the honest substitute is **end-of-schedule held-out loss**.
- **What answers it:**
  - Report the Δ(h) curve at h ∈ {50, 150, 500, end-of-schedule}, on the F1 warmup+cosine trunk.
  - Use ≥ 1M held-out tokens, identical for all branches.
  - Pre-register the *primary* horizon as end-of-schedule.
- **Cost [I]:** ≈ 3–6 P100-h (end-of-schedule forks for the primary contrast only); 3–5 person-h.
- **Before deadline?** **Yes**, if F1's trunk is short enough.

#### M12 · ROCm: a tolerance calibrated on calm steps under-states drift in the spike regime — R1
- **Objection.** improvement A6 covers the setup (hipBLASLt, AOTriton, cross-process kernel choice). What it does not
  cover [I]: V7 §4.4's fallback calibrates "5× the largest noop-vs-noop drift over 5 pairs". If those pairs run on
  calm segments, they under-state how fast a 1-ulp difference grows through a spike (M8). A tolerance-based result
  also can't be sold as "bitwise".
- **What answers it:**
  - Measure noop-vs-noop drift **across two processes, through each spike window**, before any battery.
  - If nonzero, set the tolerance from spike-window drift, drop "bitwise" from the title, and restrict claims.
  - Keep the CUDA-verified path as the reference. If ROCm fails, ask for a CUDA allocation rather than ship
    tolerance-only results (Kaggle is on hold per V7, so this needs a team decision).
- **Cost:** 3–5 GPU-h (already in V7 §4.1) + 4–8 person-h (improvement A6).
- **Before deadline?** **Yes**. It is on the critical path: nothing else is valid until it passes.

### MINOR

#### m13 · AdamW-only scope (Muon has no `v`) — R3
- **Objection.** Muon, Shampoo/SOAP and Adam-mini are mainstream in 2026. Marin's Muon swap gave "temporary relief"
  [V-prior], and AdaGC and Stable-SPAM reviewers asked about Muon (§2). Bai et al. was accepted as Adam-only, so
  this is minor.
- **Twist [I].** Standard Muon setups keep AdamW for embeddings and LayerNorm. At our proxy that is about 88% of
  the parameters, so `v` still exists for most of them.
- **What answers it:** a `torch.optim.Muon` (2-D matrices) + AdamW (rest) control. Does the damage persist in
  Muon's momentum, and does an m-only repair help?
- **Cost:** ≈ 0.4 P100-h; 6–8 person-h.
- **Before deadline?** **Yes.**

#### m14 · Embedding rows will dominate any localizer at this vocab-to-width ratio — R1
- **Objection [I].** About 88% of the parameters are tied embedding rows. Rare-token rows get sparse gradients and
  carry tiny, stale `v`, which means huge preconditioned curvature. They will rank high *regardless of the spike*.
  Wortsman 2023 already found the RMS spike in the patch-embedding layer [V-prior via RESEARCH §1].
- **What answers it:**
  - Report localization metrics separately for embedding vs block tensors.
  - Normalise by parameter count.
  - Give a per-row analysis for the embedding.
  - Optionally run an untied variant.
- **Cost:** 0 GPU-h; 3–5 person-h.
- **Before deadline?** **Yes.**

#### m15 · Too many contributions for 8 pages; essential references missing — R2
- **Objection.** C1 + C2 + C3 + theory (ψ_k, AR(2)) + conformal detection + seedable natural spikes + a benchmark is
  ≥ 6 contributions. Reviewers will find the headline claim diluted.
- **Likely "essential references not discussed" [V that each exists]:**
  - Cohen et al., adaptive EoS (2207.14484): the reference ewU8 used against Bai.
  - Takase et al., *Spike No More* (COLM 2025).
  - PaLM's rollback.
  - Molybog 2023.
  - OLMo 2.
  - Asadi 2023 · Ellis 2024 · Kalra & Barkeshli 2024.
- **What answers it:**
  - One headline claim, from F2.
  - Move detection, conformal control and the benchmark to the appendix or a separate D&B paper.
  - A related-work table built from the 16 JSONs.
- **Cost:** 0 GPU-h; 10–15 person-h.
- **Before deadline?** **Yes.**

### Summary table

| # | Objection | Severity | MUST | GPU (P100-eq h) [I] | Person-h [I] | By ~2027-01-22? |
|---|---|---|---|---|---|---|
| F1 | Proxy regime too small/early/constant-LR | fatal | **✔** | 5–9 (+20 for 124M n ≥ 3) | 15–25 | yes (124M only with follow-on hours) |
| F2 | Thesis untested; m/v self-heal in ≲ 200 steps | fatal | **✔** | ≈ 3 | 10–15 | yes |
| F3 | "Poisoned direction" sign/basis/circularity | fatal | **✔** | ≈ 0.1 | 8–12 | yes, before C2 code |
| M4 | QK-Norm / attention confound | major | **✔** | ≈ 2.5 | 6–10 | yes |
| M5 | Strawman reset; no rollback baseline | major | **✔** | ≈ 0.5 | 4–8 | yes, pre-registered |
| M6 | Novelty compression (Bai, Liu & Li, Megatron, OPEN-1B, SPAM, Spectra) | major | | ≈ 0.5 | 15–25 | yes |
| M7 | Pseudo-replicated n, MDE, multiplicity | major | | ≈ 0.7 | 6–10 | yes |
| M8 | Chaotic single-realisation Δ, no placebo | major | | ≈ 1.0 | 3–5 | yes |
| M9 | Induced spikes: wrong mode / trivial | major | | ≈ 0.3 | 6–10 | yes |
| M10 | Localization cost vs rollback | major | | 0.5–1 | 6–8 | yes |
| M11 | Horizon / end-of-schedule endpoint | major | | 3–6 | 3–5 | yes |
| M12 | ROCm tolerance vs spike-regime drift | major | | 3–5 (in V7) | 4–8 | yes (critical path) |
| m13 | AdamW-only | minor | | ≈ 0.4 | 6–8 | yes |
| m14 | Embedding-row artefact | minor | | 0 | 3–5 | yes |
| m15 | Scope + essential references | minor | | 0 | 10–15 | yes |
| | **Total (excluding 124M)** | | | **≈ 21–30** | **≈ 105–170** | |

**Budget reality check [I].**

- **GPU.** The total above, plus the existing battery, is about 25–35 P100-equivalent hours. It fits the 15 MI300X-h
  ask only if MI300X runs these shapes at ≥ ~2× P100.
- **Speed assumption.** V7 assumes 10–25×. For a 3-layer, d = 192 model at batch 16, the GPU is mostly idle on
  kernel launches, so the real speedup may be much smaller (UNVERIFIED until `step_timer`).
- **Mitigation.**
  - Run several fork processes concurrently on the 192 GB card. Whether that stays bitwise is UNVERIFIED; add it to
    the M12 gate.
  - Apply V6's cut order: 3 seeds.
- **People.** The person-hours sit on top of the V6 C2 + C3 budget (≈ 130 h) with a bus-factor-1 spike role
  (improvement C6).
- **Calendar.** Thirteen weeks to the estimated deadline is tight but feasible only if the owners in improvement §5.3
  are assigned this month.

---

## 4 · Scores for the current plan (1–10, ICML criteria)

| Criterion | Score | One-line reason |
|---|---|---|
| **Soundness** | **4** | C1 is a real, CUDA-verified asset. But the thesis isn't the primary endpoint (F2), "poison" has no sign or basis (F3), `Bg` is mis-specified (M5), n is pseudo-replicated (M7), and AMD is untested (M12). |
| **Significance** | **3** | Under 1.4M non-embedding parameters, ≤ 1 token/param, constant LR, mostly induced spikes, no QK-Norm, and damage bounded by β2's ~100–200-step memory. A reviewer can say "a 3-layer toy at step 160". |
| **Novelty** | **6** | Within-state localization + selective repair + causal verification looks unclaimed. Targeted searches on 2026-10-19 and again today found no match, but this is not a systematic review. Each ingredient has 2026 prior art (M6). |
| **Clarity** | **5** | Internally well documented and honest. But the paper has ≥ 6 contributions, and "rank-limited repair of poisoned directions of v" is not operational as written. |

**Mapped to the ICML 2026 form [I]:** Soundness 2/4 · Presentation 2–3/4 · Significance 2/4 · Originality 3/4 →
**overall 3/6 (weak reject)**, confidence ~3. With the five MUSTs done and results in either direction: **4/6
(weak accept), borderline**. That is consistent with V7 core value 6 and with the reject-then-accept path of Bai et
al. and AdaGC.

### The single change that raises the score most
**Make the w-vs-(m,v) optimizer-state patching decomposition (F2) the pre-registered primary endpoint. Measure it as
held-out loss at the end of a warmup+cosine schedule, on the proxy with and without QK-Norm.**

Why this one [I]:

- It turns the paper's thesis from assumed into tested.
- It gives C2 a causal oracle, which fixes F3's identifiability problem.
- It answers Marin directly ("optimizer state vs architecture" becomes a measured fraction, with and without
  QK-Norm).
- It makes a negative result publishable. "Spike damage is ≤ X% optimizer-state-carried at β2 = 0.95" is useful to
  practitioners and survives the Gate-B PIVOT branch.
- It costs about 3–5 P100-h and needs no harness change.

Expected effect: soundness 4 → 6, significance 3 → 5. Scale (F1) remains the ceiling after that.

---

## 5 · Limits of this review

- ICML 2027 criteria and dates are **assumed** from 2026 plus one unofficial tracker.
- OpenReview quotes are fetch-tool fragments of ≤ ~120 characters, not checked against rendered pages. ICML 2026
  reviews for Bai and AdaGC were not reachable.
- All GPU and person-hour costs are estimates from V7's P100 timings. MI300X speed is unmeasured.
- Statistics figures (MDE) are normal-approximation arithmetic, not software output.
- The OPEN-1B optimizer-state availability is UNVERIFIED.
- `arXiv 2607.06628` (cross-trajectory "chimera" weight interventions, 2026-07-07) is a possible precedent for
  state patching. Whether it transplants Adam moments is **UNVERIFIED** (the abstract doesn't say).

---

## 6 · Sources (all accessed 2026-10-22 unless noted)

**ICML process**
1. ICML 2026 Reviewer Instructions — https://icml.cc/Conferences/2026/ReviewerInstructions
2. ICML 2026 Call for Papers — https://icml.cc/Conferences/2026/CallForPapers
3. ICML 2027 CFP (404, not yet published) — https://icml.cc/Conferences/2027/CallForPapers
4. ICML 2027 estimated dates (unofficial, last reviewed 2026-10-08) — https://www.opencurious.com/ai-conference-deadlines/icml-2027

**OpenReview reviews** (via `https://api2.openreview.net/notes/search?term=…&source=reply`)
5. Bai et al., ICLR 2026 (rejected) — https://openreview.net/forum?id=dwx3rjWGju · ICML 2026 — https://openreview.net/forum?id=STWQoscanw
6. SPAM, ICLR 2025 — https://openreview.net/forum?id=L9eBxTCpQG
7. AdaGC, ICLR 2026 (rejected) — https://openreview.net/forum?id=ZQcDUhEOg9 · ICML 2026 — https://openreview.net/forum?id=uBYlCu8b8Z
8. Stable-SPAM, ICLR 2026 (withdrawn) — https://openreview.net/forum?id=BuJ2C6iZRG
9. Sharpness Disparity Principle, ICML 2025 — https://openreview.net/forum?id=DZ6iFdVDrx
10. Understanding Sharpness Dynamics (Minimalist Example), ICML 2025 — https://openreview.net/forum?id=XfjrLEPOQV
11. Small-scale proxies, ICLR 2024 — https://openreview.net/forum?id=d8w0pmvXbZ
12. Spike No More, COLM 2025 / ICLR 2025 rejected — https://openreview.net/forum?id=52YBEzcI0l · https://openreview.net/forum?id=G84F1h2IiD

**Papers and reports**
13. Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam* (ICML 2026) — https://proceedings.mlr.press/v306/bai26o.html · https://arxiv.org/abs/2506.04805
14. Liu & Li, *Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW* (2026-08-20) — https://arxiv.org/abs/2608.19762
15. OPEN-1B (2026-09-15) — https://arxiv.org/abs/2609.17380
16. NVIDIA Megatron Core bitwise-deterministic pretraining (Oct 2026) — https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/
17. Marin 32B retrospective — https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/ · issue #1368 — https://github.com/marin-community/marin/issues/1368
18. Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (ICLR 2024) — https://arxiv.org/abs/2309.14322
19. Chowdhery et al., *PaLM* (2022), training-instability section — https://arxiv.org/abs/2204.02311
20. Cohen et al., *Adaptive Gradient Methods at the Edge of Stability* — https://arxiv.org/abs/2207.14484
21. Ramasinghe et al., *Taming Curvature* (2026) — https://arxiv.org/abs/2606.16768
22. Spectra (ICML 2026) — https://arxiv.org/abs/2602.11185
23. SPAM — https://arxiv.org/abs/2501.06842 · AdaGC — https://arxiv.org/abs/2502.11034
24. Asadi et al. 2023 — https://arxiv.org/abs/2306.17833 · Ellis et al. 2024 — https://arxiv.org/abs/2412.17113 · Kalra & Barkeshli 2024 — https://arxiv.org/abs/2406.09405
25. Cross-Trajectory Chimera Interventions (2026-07-07; moment handling UNVERIFIED) — https://arxiv.org/abs/2607.06628

**Repo files read (local, uncommitted state as of 2026-10-22):** `PLAN_V7.md`, `improvement.md`,
`research/spikes/RESEARCH.md`, `research/experiments/proxy/config.yaml`, `research/baselines/reset.py`,
`research/spikes/natural.py` (lines 29–42), `research/spikes/capture.py:39`, `research/model/nanogpt.py` (LayerNorm
lines), `spike-related-work/results/*.json`.

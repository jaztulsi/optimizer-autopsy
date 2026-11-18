# Upgraded research plan: proposal for team sign-off

*Written 2026-11-12. A proposal. It does not replace `PLAN_V8.md`; where it suggests changing V8 it says so.
Nothing was run for it. Sources: V8 and its Appendix A, `docs/prereg_v2.md` and its results, `docs/novelty/gap_ledger.md`,
`docs/novelty/method.md`, and 30 web searches plus 4 page reads made today (§13).*

**Labels.** [V] checked today against a search hit or the page. [L] already in the gap ledger at the depth stated
there. [M] from memory, not re-checked. [I] my inference, worked by hand. [S] speculative. [UNV] could not be confirmed.

---

## 0. Bottom line

1. **The idea list is not the bottleneck.** V8 Appendix A already holds about 150 ideas, and seven of the eight seeds
   in the brief are in it (state patching, group testing, attribution patching for state, Muon and Adafactor
   controls, conformal detection, scaling laws). What is missing is evidence: one seed, one spike type, one small
   model, and no lasting damage at realistic dose.
2. **The brief's project description is out of date in four places.**
   - "The 124M battery runs on free Kaggle": Kaggle is paused (user instruction, Oct 2026). 124M is not funded.
   - "15 to 40 GPU-hours": the whole allocation is 20 MI300X-hours and about 14.9 remain.
   - "Planned next: chaos-floor dose sweep": prereg-v2 ran on 2026-11-10. Its frozen third outcome applies ("no
     lasting damage at realistic dose"). H-A passed; H-A2, H-B, H-C, H-E and H-F failed as written.
   - ICLR 2027 closed on 2026-09-25 [V].
3. **The original thesis does not survive as stated.** On this proxy at β2 = 0.95, a real spike's loss excess fades
   on its own (seed 0). At step 1000 it is gone within about 19 steps (0.0115 nats at the spike, 0.0012 at +19,
   −0.0001 at the end). At step 160 it fades more slowly (0.0184 at the spike, 0.0038 at +199, 0.0005 at the end).
   "Persists for thousands of steps" and "repair beats simple fixes" are not supported. "Concentrated in a few
   directions" is untested. One unexplained instance (added 2026-11-12): at step 1000 and realistic dose, undoing m
   alone left 0.117 nats at +19, ten times the spike's own excess (`rml3_method.md` §2.3).
4. **What the data do support is a different and sharper paper.** At high dose the second moment v cancels 68% to 99%
   of the damage that the first moment m would otherwise write (§2). Fixes that clear v do more harm than the
   spike. This is an analysis paper about where spike damage lives and which remedies hurt, with a repair as a
   secondary, exploratory part.
5. **Five additions from this pass** (§5): the interaction form of the mediation result (N1), a β2 data-collapse
   test (N2), a spike-gain table across optimizer families (N3), an instance-level statistical design that frees
   about 6 GPU-hours (N4), and a do-no-harm audit of remedies (N5). The core runs take 10.9 of the 14.9 hours with
   a reserve; the remaining hours are a choice between two options (§7).
6. **Venue.** Write for TMLR as the base case. Submit to ICML 2027 (abstract Jan 14, paper Jan 22 [V]) only if two
   gates are met by Dec 20 (§9).

---

## 1. Problem

**The problem.** When the loss spikes in Adam-family training, people clip, skip the batch, roll back, reset the
optimizer or cut the learning rate. Nobody has measured what the spike left behind against a run that saw the same
data without it (ledger gap G5). So the remedy is chosen without knowing whether there is anything to remedy, or
where it is stored.

**Why current remedies are inadequate.**
- They are judged by final loss across different runs, not against a same-data counterfactual [L, G5].
- State resets can cost more than the spike. On seed 0, SPAM and a v reset cost 0.05 to 0.08 nats at steps 160 and
  1000, and the v reset cost 1.6 nats at step 1800, on a run whose spike left at most 0.0024 nats [V, our data].
  SPAM cost only 0.004 nats over doing nothing at step 1800, which hints that its cost follows its reset schedule
  [I, not checked].
- SPAM reports no numeric ablation that separates its clip from its reset [L, G13].
- Rollback is exact but expensive (PaLM restarts about 100 steps back and skips 200 to 500 batches [L]).
- No optimizer-side rule tells a one-off outlier from a real change or an instability [L, G6].

**Who cares.** Teams that pretrain language models and report spikes with ad-hoc rules (OLMo 2, PaLM [L]);
optimizer designers; people working on reproducible training.

**What would count as a real result, in order of how well the evidence can support it.**
1. A replicated answer to "does a spike leave lasting damage, and in which part of the state", with a law that says
   when (dose, β2, timing).
2. A safety result: which common remedies do harm, and the mechanism.
3. A repair that beats oracle skip-step in a regime where damage exists. This is not shown and may not exist.

---

## 2. What seed 0 already shows when read as a 2 × 2 (exploratory, no GPU)

The four undo arms at lag 0 are a full factorial over "is the spike's share left in m" and "is it left in v".
Write Y(a_m, a_v) for the end-of-run damage in nats, with 1 = left in and 0 = undone. Then

- direct write D = Y(0,0) (undo-m+v);
- m alone E_m = Y(1,0) − Y(0,0) (undo-v minus undo-m+v);
- v alone E_v = Y(0,1) − Y(0,0) (undo-m minus undo-m+v);
- interaction I = Y(1,1) − Y(1,0) − Y(0,1) + Y(0,0);
- total Y(1,1) = D + E_m + E_v + I (exact identity);
- protection share π = −I / E_m: the fraction of m's bare damage that v's inflation cancels.

This is VanderWeele's four-part split of an effect under mediation with interaction [V], which V8's research notes
already name (`02_adjacent_fields.md`, T2). The numbers are new.

| Step | c | D | E_m | E_v | I | Total | π |
|---|---|---|---|---|---|---|---|
| 160 | 10 | 0.0021 | 0.0453 | 0.0022 | −0.0419 | 0.0078 | 0.92 |
| 160 | 30 | 0.0023 | 0.5982 | 0.0118 | −0.5874 | 0.0250 | 0.98 |
| 1000 | 10 | 0.0018 | 0.0238 | 0.0019 | −0.0161 | 0.0114 | 0.68 |
| 1000 | 30 | 0.0011 | 0.2019 | 0.0130 | −0.2006 | 0.0154 | 0.99 |
| 1800 | 10 | −0.0003 | 0.0453 | 0.0062 | −0.0402 | 0.0110 | 0.89 |
| 1800 | 30 | −0.0006 | 0.5725 | 0.0120 | −0.5595 | 0.0244 | 0.98 |

Source: `results/mi300x_2026-11-10/.../report_final/sealed.json`, seed 0, one run per cell. Doses c ≤ 3 are left out:
their cells sit inside the run-to-run scatter (§6.5).

**Readings [I].**
- m alone grows like the square of the dose: log-log slopes 2.35, 1.95 and 2.31 between c = 10 and 30. A displacement
  linear in dose and a loss quadratic in displacement give exactly 2.
- The total grows slower than the dose (slopes 1.06, 0.28, 0.72) because π rises with dose in every phase.
- v alone (the stall) is small, 0.002 to 0.013 nats, with slopes from 0.6 to 1.7.
- This explains why blind prediction B3 failed (`method.md` §6.1a). The registered "echo" was none minus undo-m,
  which equals E_m + I: the net echo after protection. It falls when π rises. B3 stays scored as failed.
- **One result goes against the hand theory.** At step 160 and c = 10, π is 0.88, 0.92 and 0.97 for β1 = 0.8, 0.9 and
  0.95 (three different trunks, no placebos). The frozen-noise model predicts the opposite order once the spike is
  above the knee, because the knee D rises with β1 (4.99, 5.84, 8.83). This is an open discrepancy to test, not to
  explain away.

Everything in this section came from looking at seed 0. It can only become a claim through a new pre-registration
tested on seeds that have not been looked at (§6.7).

---

## 3. Related work

Depth is stated per row. Rows marked [L] are in the gap ledger with their own depth notes (78 entries there).

| Area | Paper | Key idea | Relation to this project | Check |
|---|---|---|---|---|
| Spike theory | Molybog et al. 2023, [2304.09871](https://arxiv.org/abs/2304.09871) | Adam instability at scale from correlated updates | Why spikes start; we study what they leave | [L] |
| Spike theory | Bai et al., ICML 2026, [2506.04805](https://arxiv.org/abs/2506.04805) | v decays while gradients are small, the preconditioned curvature crosses the stability limit, five-stage spike | Mechanism of spikes that build up; its β2 timescale is the one our collapse test uses | [V] |
| Spike theory | Tang, Chen, Liu 2026, [2609.18314](https://arxiv.org/abs/2609.18314) | Stability phase diagram in (β1, β2); uses deterministic replay to rebuild pre-spike state | Nearest rival to the instrument. Title and id confirmed today, which settles the retrieval doubt in `method.md` §10 | [V] title and id; the replay detail is from a snippet |
| Spike theory | Kim 2026, [2608.15824](https://arxiv.org/abs/2608.15824); Patitucci & Mokhtari 2026, [2602.10204](https://arxiv.org/abs/2602.10204) | v's long memory of a rare large gradient; per-update response to one spike | Qualitative stall; no cumulative bound, no experiment that varies β2 | [L] |
| Spike theory | Wortsman et al. 2023, [2309.14322](https://arxiv.org/abs/2309.14322) | Small models reproduce large-model instabilities | The licence for a proxy study, and its limit | [L] |
| Spike control | ICLR 2027 submissions [pAjrfibDB9](https://openreview.net/forum?id=pAjrfibDB9), [P0oFJZXAh8](https://openreview.net/forum?id=P0oFJZXAh8) | Limiting coordinates with large m/√v; mismatched m and v histories | Prior art for selectivity and for joint m/v effects | [L], abstracts only |
| Edge of stability | Cohen et al. 2021, [2103.00065](https://arxiv.org/abs/2103.00065); Cohen et al. 2022, [2207.14484](https://arxiv.org/abs/2207.14484) | Sharpness hovers at the stability limit; 38/η for Adam | Source of the stable band used in the theory | [M] / [L] |
| Edge of stability | Cohen, Damian et al., ICLR 2025, [2410.24206](https://arxiv.org/abs/2410.24206) | "Central flow": an ODE for the time-averaged path of an oscillating optimizer, including RMSProp | The right tool for a better post-spike theory than our frozen model | [V] |
| Curvature tools | Yao et al. 2020, PyHessian, [1912.07145](https://arxiv.org/abs/1912.07145); curvlinops | Hessian-vector products, Lanczos | Tooling for C2. C2 is proposed for the cut list | [M] |
| Curvature tools | Yu, Wang, Samworth 2015, [doi 10.1093/biomet/asv008](https://doi.org/10.1093/biomet/asv008) | Davis–Kahan bound that needs only the population eigengap | Bound on how far a localized subspace rotates (§8, T6) | [V] |
| Baselines | SPAM [2501.06842](https://arxiv.org/abs/2501.06842); ZClip [2504.02507](https://arxiv.org/abs/2504.02507); AdaGC [2502.11034](https://arxiv.org/abs/2502.11034); StableAdamW [2304.13013](https://arxiv.org/abs/2304.13013) | Clip rules and periodic resets | Comparators; SPAM's reset is the one our data flag as harmful | [L] |
| Baselines | Asadi et al. 2023, [2306.17833](https://arxiv.org/abs/2306.17833) | Resetting v alone helps deep RL at β2 = 0.999 | Opposite sign to our v-reset result, in a different setting. Must be discussed, not ignored | [L], snippet today |
| Mediation | VanderWeele 2014, [doi 10.1097/EDE.0000000000000121](https://doi.org/10.1097/EDE.0000000000000121) | Four-part split of an effect under mediation and interaction | The estimand of §2 | [V] |
| Patching | Heimersheim & Nanda 2024, [2404.15255](https://arxiv.org/abs/2404.15255) | Noising and denoising are not symmetric; use several metrics | Design rules for state patching: report both directions | [V] |
| Patching | Kramár et al. 2024, AtP\*, [2403.00745](https://arxiv.org/abs/2403.00745) | Gradient approximation to patching, with known false negatives | Model for a cheap localizer and for its failure modes | [V] |
| State surgery | Xu 2026, [2608.20442](https://arxiv.org/abs/2608.20442) | Transplants m (or weights plus m) between runs; a full-horizon costate predicts the sign of the effect | **Takes** state patching and the adjoint as methods. Its setting is trait transfer, not spikes. The ledger's partial read has v carrying about 0% of the effect there | [V], abstract via a page reader |
| State maps | Sweeney 2026, [2606.30788](https://arxiv.org/abs/2606.30788) | Treats an AdamW step as a map over (w, m, v); second-order proof for a weight-space "revocation" edit | Nearest formal treatment of removing a past contribution; edits weights, not moments | [V], abstract via a page reader |
| Replay | Engstrom et al. 2025, [2503.13751](https://arxiv.org/abs/2503.13751) | Exact gradients through whole training runs by deterministic replay | Tool for a costate localizer (N6) | [V] |
| Reproducibility | Zhuang et al. 2021, [2106.11872](https://arxiv.org/abs/2106.11872) | Tooling noise; deterministic training costs up to 746% overhead on some GPUs | Context for the determinism gate | [V] |
| Reproducibility | Kwok et al., ICML 2025, [2506.13234](https://arxiv.org/abs/2506.13234) | One perturbed weight early in training separates two otherwise identical runs | **Takes** the qualitative chaos-floor finding (our floor falls from 1e-4 to 3e-8 with training) | [V] id and abstract summary |
| Reliability | Ma et al. 2025, [2502.12340](https://arxiv.org/abs/2502.12340); TrainSDC, [2608.30769](https://arxiv.org/abs/2608.30769) | Deterministic execution to compare healthy and faulty nodes; faults can cause loss spikes | Rival to the instrument in the hardware-fault setting | [V] |
| Other optimizers | Lion [2302.06675](https://arxiv.org/abs/2302.06675); SOAP [2409.11321](https://arxiv.org/abs/2409.11321); Muon at scale [2502.16982](https://arxiv.org/abs/2502.16982); Adafactor [1804.04235](https://arxiv.org/abs/1804.04235) | Sign momentum; Adam in Shampoo's eigenbasis; orthogonalized momentum; factored v | Families for the spike-gain table (N3) | [V] / [L] |
| Other optimizers | ROOT [2511.20626](https://arxiv.org/abs/2511.20626); Musec [2609.11655](https://arxiv.org/abs/2609.11655); TrasMuon [2602.13498](https://arxiv.org/abs/2602.13498) | Muon is not robust to outliers and has its own spikes | Shows the question is not Adam-only | search snippets only |
| Early warning | Scheffer et al. 2009, [doi 10.1038/nature08227](https://doi.org/10.1038/nature08227); Quail, Shrier, Glass 2015, [doi 10.1073/pnas.1424320112](https://doi.org/10.1073/pnas.1424320112) | Rising autocorrelation and variance before a transition; for period doubling the lag-1 autocorrelation equals the return-map slope | The thermometer in `method.md` §2.6 is this statistic; the closed form for Adam's mode is ours | [V] |
| Early warning | Kim, TIC, ICML 2026 workshop, [link](https://icml.cc/virtual/2026/80057); Hennick & Corlouer 2026, [2603.29805](https://arxiv.org/abs/2603.29805) | Hazard model that forecasts spikes; early warning of phase transitions in training | Spike forecasting is taken | [L] / snippet only |
| Stability proofs | Lessard, Recht, Packard 2016, [1408.3595](https://arxiv.org/abs/1408.3595) | Optimizers as a linear system in feedback with a bounded nonlinearity; stability by a small semidefinite program | A route to a self-healing proof without assuming H and D commute | [V] |
| Statistics | Lakens 2017, [doi 10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177); Agarwal et al. 2021, [2108.13264](https://arxiv.org/abs/2108.13264); Bhattacharjee & Seno 2001, [doi 10.1088/0305-4470/34/33/302](https://doi.org/10.1088/0305-4470/34/33/302) | Equivalence tests; interval estimates with few runs; a numeric measure of data collapse | Tests for "no lasting damage", few-seed reporting, and N2 | [V] |

---

## 4. Open gaps, and which ones this project can hit

| # | Gap | Evidence | Can we hit it with 14.9 h? |
|---|---|---|---|
| O1 | Lasting effect of a spike against a same-data counterfactual | Ledger G5 | Yes. Seed 0 done; replication is E1 |
| O2 | When damage exists: no experiment varies β2 and fits recovery | One search today found none | Yes: E2 |
| O3 | Total displacement from one outlier, compared across optimizer families | One search found robustness rates and Muon fixes, no per-event measurement | Theory yes; one small experiment (E4) |
| O4 | Which part of SPAM does what; whether v resets help or harm | Ledger G13; Asadi 2023 against our seed 0 | Yes: three extra arms in E1 |
| O5 | Telling an outlier from a real change or an instability | Ledger G6 | Only as an exploratory arm where damage exists |
| O6 | Early warning for spikes that build up | Forecasting taken by TIC; no hit for lag-1 autocorrelation at the stability edge | Optional, low value for bad-batch spikes |
| O7 | Natural spikes with captured state at language-model scale | Ledger G11; V8 K8 | No. Needs hours or released states |
| O8 | Transfer across model size | V8 K10 | No. 124M is unfunded |
| O9 | Whether effects (not trajectories) reproduce across CUDA and ROCm | None found | No while Kaggle is paused |
| O10 | Error from treating curvature modes separately | Seed (d) | Only if C2 is kept |

---

## 5. Method proposals

"In V8" gives the Appendix A id where the idea already exists. GPU-hours here are rough first estimates. The
experiment budget in §7 uses a better basis (fork-steps) and is the one to trust.

| ID | Idea | In V8? | Closest prior work | Novelty verdict | Benefit | GPU-h | Difficulty | Main risk |
|---|---|---|---|---|---|---|---|---|
| N1 | Four-part split with protection share π as the primary estimand (§2) | C-2 has the 2 × 2; T2 names the split | VanderWeele 2014; Xu 2026 | New as a finding. Not new as a method | Turns three failed hypotheses into one coherent, testable account | 0 now; replication in E1 | Low | One seed |
| N2 | β2 data collapse: π and recovery time against x = (1−β2)z², time in units of 1/(1−β2) | Related to E2, X11 | Kim 2026; Bai 2026 | No experiment found that varies β2 and fits recovery (one search) | Says when repair matters; the default β2 = 0.999 is where the stall should be 50 times longer | 2.5 | Medium | No collapse; then report per-β2 laws |
| N3 | Spike gain: total displacement from one outlier, by optimizer family (theory table, one measurement) | #20, R9, R10 in part | Patitucci & Mokhtari 2026; ROOT | Likely new as a per-event quantity (one search) | Answers "is this Adam-only?"; a robustness index others can reuse | 0 theory; 0.8 for Lion | Medium | Frozen model ignores the restoring gradient |
| N4 | Instance-level design: effect scatter, not the chaos floor, sets the noise; floor measured as a curve over training | Pieces in §6.9 | Kwok 2025; Agarwal 2021 | Methodology; no novelty claim | Frees about 6 h; fixes the "ruler problem" in the results note | 0.7 | Low | None |
| N5 | Do-no-harm audit: SPAM whole, clip-only, reset-only; v reset; cooldown | #8 | SPAM; Asadi 2023 | New as a measurement | A result practitioners can act on today | 0.5, inside E1 | Low | Harm may be proxy-specific |
| N6 | Costate localizer: gradient of final loss with respect to (w, m, v) at the spike, by reverse replay, scored against forks | #11, #16 in part | Xu 2026; Engstrom 2025; Sweeney 2026 | **Partial.** The costate exists; scoring it as a spike localizer is the delta | Replaces Lanczos for C2 with one backward pass | 1.5 to 2 | High | Nothing to localize at realistic dose |
| N7 | Probation (RML-2): withhold, test on later gradients, then keep or skip; tested only where damage exists | `method.md` | Swift; TAdam; WoLF; gated fixed-lag smoothers in navigation ([2109.14220](https://arxiv.org/abs/2109.14220), snippet only) | Partial. Its arming threshold is unvalidated | A repair with a stated false-alarm cost | 1.5 | Medium | May only tie oracle skip-step |
| N8 | Early warning for spikes that build up: lag-1 autocorrelation falling toward −1, alarm rate set by conformal calibration | #60, #61 | TIC; Quail 2015; Scheffer 2009 | Partial. Forecasting is taken | A cheap free signal | 0 to 1 | Low | Useless for bad-batch spikes |
| N9 | Davis–Kahan bound on subspace rotation with a measured perturbation | no | Yu et al. 2015 | New only as an application | Rigor for C2 | 2 to 4 | High | Depends on C2, which is on the cut list |
| N10 | Cross-hardware reproducibility of effects | X8 in part | Zhuang 2021; Ma 2025 | Unknown | A contribution of its own | Needs CUDA | Medium | Blocked while Kaggle is paused |

**How the brief's seeds map.** (a) state patching: taken as a method by Xu 2026 and already V8's K3/K4; what is ours
is N1. (b) other optimizers: N3. (c) early warning: N8. (d) Davis–Kahan: N9. (e) cheap localizers: N6 and V8's no-HVP
localizers. (f) minimal sufficient repair: V8 #6 and #43; kept as a concentration curve inside the stretch. (g) chaos
floor and cross-hardware: N4 and N10. (h) scaling with β2: N2; scaling with model size is unfunded.

---

## 6. Final methodology (N1 + N2 + N5, with N4 as the design and N3 as support)

### 6.1 Definitions
- **Instance** i = (seed, injection step, spike source). Each instance has its own data-aligned clean twin B\*.
- **Outcome** Y_i(arm) = held-out loss of the arm at horizon T minus B\*'s, in nats.
- **Primary estimands** D, E_m, E_v, I and π as in §2; the log-log dose slope of E_m and of the total.
- **Recovery time** t_rec: the first evaluated horizon after which |Y(none)| stays below 0.005 nats.
- **Dominance** x_i = (1−β2)(g_i − m_i)²/v_i per coordinate. Log its 10%, 50% and 90% quantiles on the hit support,
  not only the mean that the report stores now. (A logging change in the autopsy code; `determinism.py` and
  `preflight.py` are not touched.)
- **Spike gain** G = displacement of the hit coordinates against B\* at horizon k, divided by the learning rate.

### 6.2 Procedure
1. From the snapshot at the injection step, fork the 2 × 2 undo arms at lag 0, at doses c ∈ {1, 10, 30}.
2. At c = 1 add the remedy arms: oracle skip-step, detector-timed skip-step, RML at lag 3, SPAM, SPAM with clip only,
   SPAM with reset only, v reset.
3. Add 8 placebo forks (1-ulp) per injection step.
4. For N2, train trunks at β2 ∈ {0.95, 0.99, 0.999}. Inject so that the end of the run is one v-memory, 1/(1−β2)
   steps, after the spike (steps 1980, 1900 and 1000). Scale the doses by √(0.05/(1−β2)) and check the match on the
   logged dose x̄.
5. Compute the four-part split per instance, then aggregate over seeds.

### 6.3 Data, models, baselines
The proxy and data in `research/experiments/proxy/config.yaml`; spike sources `corrupt_batch` (main) and `lr_bump`
(one check, E5). Baselines are V8 §6.7's, each tuned on seed 0 only, plus the two split SPAM arms.

### 6.4 Ablations, one per claim
Dose (1, 10, 30); timing (steps 1000 and 1800); β2 (three values); β1 (0.8, 0.9, 0.95, for the discrepancy in §2);
spike source (bad batch against learning-rate bump); lag (0, and 3 and 10 at β2 = 0.999 in the stretch).

### 6.5 Statistics
- **Unit.** The seed. Injection steps within a seed share a trunk and are not independent.
- **Two noise scales.** The placebo floor is about 4e-7 nats at step 1000, so each instance's effect is measured
  almost without error. Effects still differ in sign and size between instances by a few thousandths of a nat (at
  c = 1 and step 1000: none −0.0001, undo-m +0.0050, undo-m+v +0.0026). The second scale is the one that matters for
  inference, and only more seeds estimate it.
- **Equivalence.** "No lasting damage at realistic dose" is a two one-sided test against ±0.01 nats, the prereg's
  practical size [Lakens 2017].
- **Directional tests.** Exact sign-flip tests over seeds. With n seeds the smallest one-sided p is 1/2^n: 0.031 at
  n = 5 and 0.0078 at n = 7. The hypotheses form two families (realistic dose; high-dose mechanism), each tested in
  a fixed order at α = 0.025. So **n = 7 new seeds** is the smallest design in which the exact test can pass, and it
  still passes if one seed is weak. With 5 seeds it cannot.
- **Effect sizes.** Mean and bootstrap interval per estimand; every seed's value shown.
- **Collapse.** With three β2 values and two matched doses, the registered test is simple: π at matched x̄ differs
  by less than 0.15 between β2 values. The Bhattacharjee–Seno measure is reported beside it, descriptively.

### 6.6 Hypotheses to register (proposals; thresholds to be frozen before any new seed)

| # | Statement | Refuted if |
|---|---|---|
| P1 | At c = 1, mean Y(none) is within ±0.01 nats at steps 1000 and 1800 | the equivalence test fails |
| P2 | At c ≥ 10, Y(undo-v) > Y(none) | the sign-flip test fails |
| P3 | π ≥ 0.5 at c = 10, and π(30) > π(10) | either fails |
| P4 | The dose slope of E_m is between 1.5 and 2.5, and the slope of the total is below 1 | either is outside |
| P5 | At c = 1, v reset costs more than 0.01 nats over doing nothing at both steps, and SPAM's reset-only arm costs more than its clip-only arm | either fails |
| S1 (secondary) | One v-memory after the spike, π at matched x̄ differs by less than 0.15 across β2 | it differs by more |
| S2 (secondary) | π falls as β1 rises (the hand theory) | π rises with β1, as on seed 0 |

### 6.7 Pre-registration plan
- Seed 0 formed these hypotheses, so it is exploratory for all of them and is excluded from every test above.
- The frozen prereg-v2 hypotheses are still scored as written on seeds 1 to 4, "at the chosen doses" (its own
  wording). Their failures on seed 0 stay in the record.
- `docs/prereg_v3.md` is drafted (2026-11-12, unsigned). It holds P1 to P5, S1, S2, the per-seed statistics, the
  seed count, the battery and the run commands, and it is the wording to trust where this section differs. It is
  tagged before the first new seed runs. Seeds 5 to 7 are an extension and are declared there.
- No threshold moves after tagging. Deviations are logged the way v2 logged its clock cut.

---

## 7. Experiment plan against the 14.9 hours

Basis: about 100,000 fork-steps per GPU-hour, my reconstruction of the 2.1-hour v2 session (an upper bound on cost,
since that session also ran gates, tests and restarts). Forks from step 160 run 1,840 steps and forks from step 1800
run 200, so early injections cost about nine times more. The same table, with the commands, is in `docs/prereg_v3.md`.

| ID | What | Fork-steps | GPU-h [estimate] | Value per hour |
|---|---|---|---|---|
| E0 | Reproduce §2 in the analysis code; wire the two split SPAM arms | 0 | 0 | highest |
| E1 | Replication: 7 seeds × steps 1000 and 1800 × (3 doses × 5 arms, 8 placebos, the remedy battery with the two split SPAM arms) | 430,000 | 4.3 | 1 |
| E2 | β2 regime on 3 seeds: 0.95, 0.99 and 0.999 one v-memory before the end, 0.99 also at step 1000, and the remedy battery at 0.999 | 260,000 | 2.6 | 2 |
| E2b | β1 arms (0.8, 0.95) on 3 seeds at step 1800, for the discrepancy in §2 | 46,000 | 0.5 | 3 |
| E3 | Floor as a curve: 8 placebos at 6 points of training | 48,000 | 0.5 | 4 |
| Reserve | Reruns | | 3.0 | |
| **Subtotal** | | | **10.9** | |
| Then either | Extension A: step 160 on seeds 1 to 4, which lets v2's hypotheses be scored in all three phases | 340,000 | 3.4 | |
| or | Exploratory set: `lr_bump` check (E5, 0.4), Lion spike gain (E4, 0.8), lag sweep at β2 = 0.999 (1.0) | | 2.2 | |

The two options do not both fit in 14.9 hours. Which one runs is a team decision (prereg-v3, open decision 2).

- **Suggested change to V8 §11.2.** The K = 111 chaos floor (7 to 8 h in the week of Nov 23) is replaced by E3. The
  floor is already so far below the practical size that a tighter estimate of it buys no inference. This is a
  suggestion for the team, not a decision.
- **Measure before committing.** Run seed 1 of E1 first and read its real cost. If seven seeds would cost more than
  6 hours, run five and use t-tests as the primary test (the cost rule in prereg-v3).
- **E4 needs code first:** a Lion optimizer and a passing `Δ==0` gate for it, tested in CI. Lion keeps only
  `exp_avg`, so `snapshot.py` probably needs a small, flagged change to accept a missing `exp_avg_sq` (V8 §9.4 lists a
  similar one for Muon). Not checked against the code.
- **Not funded, flagged as overage if wanted:** a 124M rung, Pythia states, natural spikes, a CUDA arm.
- Every session still needs Jasraj's go-ahead (CLAUDE.md §5).

---

## 8. Theory plan

All statements are for the frozen-noise model unless said otherwise: one coordinate, steady pre-spike scale √v̄, one
outlier of size z in those units, later gradients not reacting to the displacement. That last assumption is the
weak one, and it makes every bound an upper bound.

| # | Statement | Status | Test |
|---|---|---|---|
| T1 | **Linear-state lemma.** If an optimizer's state follows s_t = A s_{t−1} + B φ(g_t), the contribution of step s can be removed exactly at any later lag from a stored φ(g_s). Covers Adam's m and v, Lion's momentum, Adafactor's row and column sums, the Shampoo and SOAP factor averages, Muon's momentum | [I], follows from linearity. Not a novelty claim alone (Swift, undo logs) | unit test in CI |
| T2 | **Echo bound.** Adam's total displacement from one outlier is at most η·min(z, D), D = (1−β1)/(√(1−β2)(1−r)), r = β1/√β2 | [I], re-derived by the round-3 panel | E4, in displacement |
| T3 | **Interaction theorem (to prove).** With a quadratic loss, E_m ∝ z², the total saturates, and π = 1 − O(D²/z²) | [I], sketch only | P3, P4; S2 tests its β1 prediction, which seed 0 contradicts |
| T4 | **Stall law.** Steps lost = (2/(1−β2))·ln((1+√(1+x))/2) | [I], continuous limit | S1 |
| T5 | **Spike-gain table** (below) | [I] for SGD and Adam, [S] for Lion and Muon | E4 for Lion |
| T6 | **Subspace staleness.** For the top-k eigenspace of the preconditioned Hessian at the spike and L steps later, ‖sin Θ‖ ≤ 2·min(√k‖E‖op, ‖E‖F)/gap_k, with E the measured change. Says whether a localization is still valid when the repair fires | Yu et al. 2015 gives the bound; the application is ours | only if C2 is kept |
| T7 | Replace the frozen model by a central flow for the post-spike mean path | [S] | future work |

**Spike gain, in units of each optimizer's own step [I/S, by hand, not measured].**

| Family | State | What caps one outlier | Gain for an outlier of size z |
|---|---|---|---|
| SGD with momentum | m | nothing | grows like z |
| AdamW | m, v | v's inflation | at most min(z, D): 5.84 at (0.9, 0.95), 31.8 at (0.9, 0.999) |
| AdamW with v cleaned | m | nothing | z. This is the undo-v arm |
| Lion | m | the sign | about ln(z/z0)/(1−β2) once z > z0 ≈ 11 at (0.9, 0.99): near 100 steps at z = 30 |
| Muon | momentum | orthogonalization caps each direction | about ln(z/z0)/(1−μ) per hit direction [S] |
| Adafactor | row and column sums | v's inflation, but smeared | as Adam at the hit entry; the stall spreads to its whole row and column |

The table's one sharp claim: only a second moment with memory makes the total displacement independent of the
outlier's size, and cleaning it removes exactly that property. Lion's defaults are from memory of its paper [M].

---

## 9. Conference fit

**Dates.** ICML 2027: abstract Jan 14, full paper Jan 22, decisions May 3, conference Jul 31 to Aug 6 (from
`icml.cc/Conferences/2027/Dates`, read through a page reader; the city is not on that page) [V]. V8 §4 says
"abstract ≈ Jan 16"; that should be corrected. ICLR 2027: closed Sep 25, 2026 [V]. NeurIPS 2027: no call yet;
trackers project mid to late May [UNV]. TMLR: rolling; it accepts work whose claims are supported by evidence and
that some audience would want, even if the contribution is modest [V].

**What reviewers would praise.** Bitwise counterfactual forks with a placebo floor. A pre-registration whose
failures are reported. A mechanism that runs against common practice (clearing v is the harmful move). A harm
audit of named remedies.

**What they would attack, and what answers it.**

| Attack | Answer in the plan | Still open |
|---|---|---|
| A tiny model; why would this hold at scale | State the scope; the β2 law is the part most likely to transfer | 124M is unfunded |
| A planted, amplified spike is not a real one | c = 1 is the unamplified bad batch; the exploratory set adds a learning-rate bump if it runs | No natural spikes |
| One seed | E1 | |
| No damage at realistic dose, so who cares | That is finding 1; N2 says where damage should exist | β2 = 0.999 may also show none |
| Every component has precedent (Xu 2026 and others) | V8 §5 already concedes this; the claim is the study and its findings | |
| Hypotheses failed and new ones were written after | Seed 0 is excluded; v3 is tagged before new data | |
| A v reset helps in RL (Asadi 2023) | Discuss it: different setting, target shifts, β2 = 0.999 | E2 at 0.999 may agree with them |
| The theory is a toy | Three predictions with one already contradicted, reported as such | T3 is unproved |

**Claims and the evidence each needs.**

| Claim | Minimum evidence |
|---|---|
| A real bad batch leaves no lasting damage here | P1 on 7 seeds |
| v cancels most of m's damage at high dose | P2, P3 on 7 seeds |
| The dose law | P4 |
| Resets do harm | P5, with the split SPAM arms |
| The regime is set by (1−β2)z² | S1 |
| A repair helps where damage exists | A repair arm beating oracle skip-step at β2 = 0.999. Not expected with confidence |

**Honest odds (my judgment, not data).** ICML main track with E1 to E3 on a proxy only: low. Work on language-model
training dynamics is usually expected to show at least one model of 100M parameters or more. TMLR: good if the
claims are scoped to what was measured. An ICML or NeurIPS 2027 workshop: high.

**Recommendation.**
- **Framing:** an analysis paper. Working title: "What a loss spike leaves in Adam's state: a deterministic
  counterfactual study". Not a method paper and not a benchmark.
- **Base case:** TMLR, submitted when E1 and E2 are in and the draft is clean.
- **ICML on Jan 22 only if, by Dec 20,** (a) P2 to P5 pass on the new seeds and (b) E2 shows either a collapse or a
  regime with lasting damage at c = 1. The same paper cannot be under review at both.
- **If hours arrive:** one 124M confirmation of P2 and P3 is the single run that most improves main-track odds.
- **If the localizer or repair underperforms:** nothing changes. The paper above does not depend on them. C2
  (Lanczos, HVPs) and the rank-limited C3 move to future work; V8 already notes that no confirmatory test needs
  the HVP pieces.

---

## 10. Risks and cut order

**Cut first to last:** N9 and the Lanczos side of C2 → N6 costate → N8 early warning → E4 Lion → E3 → E5. E1 and E2
are never cut.

| Risk | What happens then |
|---|---|
| Seeds do not replicate the protection result | Report it. The paper becomes the instrument plus a null, for a workshop or TMLR |
| The β2 = 0.999 trunk trains differently | Match instances by loss level, not by step, and say so in the prereg |
| A seed costs more than estimated | Drop to 5 seeds (§7) |
| GPU access ends early | E1 alone still supports claims 1, 2 and 4 |
| Lion's determinism gate fails | Drop E4; keep T5 as stated theory |
| The team has ten weeks | The cut order above; no new modules beyond Lion and the analysis code |

---

## 11. Timeline to Jan 22, 2027

| Week | GPU | Work |
|---|---|---|
| Nov 12–15 | 0 | Team reads this. Decide N1 to N5. Draft `prereg_v3.md`. Analysis code for the four-part split, tested in CI |
| Nov 16–22 | about 2 h | Tag v3. E1 seed 1 (measure cost), then seeds 2 and 3 |
| Nov 23–29 | about 3 h | E1 seeds 4 to 7. E3 |
| Nov 30–Dec 6 | about 3 h | E2 and the β1 arms |
| Dec 7–13 | 2.2 to 3.4 h | Extension A, or the exploratory set. Statistics on real results |
| Dec 14–20 | 0 | **Gate: ICML or TMLR.** Draft 1 |
| Dec 21–31 | reserve | Draft 2; every number traced to a results file |
| Jan 1–14 | 0 | Reviewer simulation; fixes; ICML abstract on Jan 14 if the gate passed |
| Jan 15–22 | 0 | Submit to ICML, or finish for TMLR |

---

## 12. What this pass did not do

- No full texts were read today. New entries are at abstract or snippet depth, as marked.
- Three sources were seen only as search snippets (ROOT, Musec, TrasMuon), and one arXiv hit on sign methods under
  heavy tails was left out because its title was not confirmed.
- "No hit" claims rest on one search each (N2, N3, the lag-1 autocorrelation gap, fixed-lag smoothing for
  optimizers). They bound nothing beyond that search.
- The round-1 adjacent-fields house from the earlier council never ran; this pass covers part of what it would have.
- GPU-hour figures are estimates from one session's ledger.
- The cost and feasibility of Lion in the harness were not checked against the code.

## 13. Search log

Searches (30): venue dates for ICML and NeurIPS; metagradient replay; activation patching guidance; AtP\*; four-way
decomposition; silent data corruption; butterfly effect; period-doubling early warning; early-warning review;
central flows; state transplant; costate and state sensitivity; Muon instability; Lion; Muon at scale; SOAP;
Davis–Kahan variant; integral quadratic constraints; tooling randomness; early warning in training; fixed-lag
smoothing with gating; recovery time against β2; per-event displacement across optimizers; protective second
moment; recent spike-repair preprints (extended search); data-collapse measure; few-run statistics; equivalence
tests; TMLR criteria. Pages read: the ICLR 2027 call, the ICML 2027 dates page, and the arXiv abstracts of
2608.20442 and 2606.30788.

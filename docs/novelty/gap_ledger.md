# Gap ledger: loss-spike repair in Adam-style optimizers

Written 2026-11-10. This is the reading record behind `method.md`. It lists what each paper acts on, where its
analysis stops, and the gaps that follow. Per-paper notes (rules, sections, limits) are in
`master-research/raw/novelty-2026-11/results/c*.md`.

## How to read this
- **Depth** is what was actually read in this run: *full*, *partial* (HTML text, usually the first 100k characters),
  *abstract*, or *not opened*. Nothing marked abstract or not opened supports a claim here.
- **established** = stated in a paper that was read. **inference** = our reading of it. **unverified** = could not be
  checked.
- Reading tools returned page text, sometimes summarised. Formulas quoted in the cluster files should be re-checked
  against the PDF before they go into a paper.
- 78 entries: 59 read in part or in full, 14 abstract only, 5 not opened or not found.

## 1. Papers by cluster

### Spike mechanisms (c01)
| Paper | What it says or acts on | After-the-fact repair? | Depth |
|---|---|---|---|
| Molybog et al. 2023, *A Theory on Adam Instability in Large-Scale ML*, [2304.09871](https://arxiv.org/abs/2304.09871) | Time-correlated updates when gradients shrink toward ε; skipping the trigger batch breaks the chain | no | partial |
| Bai et al. 2026, *Adaptive Preconditioners Trigger Loss Spikes in Adam*, [2506.04805](https://arxiv.org/abs/2506.04805) | v decays, preconditioned sharpness passes 2/η, gradients grow, v refills, spike ends. Fix: lower β2, raise ε | no; lasting damage not addressed in the part read | partial |
| Tang, Chen, Liu 2026, *Beyond Quadratic Loss*, [2609.18314](https://arxiv.org/abs/2609.18314) | Stability phase diagram in (1−β1, 1−β2); critical step 2(1+β1)/((1−β1)λ) | no | partial |
| Li, Zhou, Xu 2026, *Weight-norm Criticality*, [2607.21005](https://arxiv.org/abs/2607.21005) | Spikes from weight decay shrinking scale-invariant weights; no v involved | no | partial |
| Tian, Paquette 2026, *Simulating Training Spikes ... Random Feature Models*, [2610.13524](https://arxiv.org/abs/2610.13524) | Heavy-tailed spectra and drift into large-eigenvalue directions (SGD) | no | partial |
| Zhang, Xu, Li 2023, *Loss Spike in Training Neural Networks*, [2305.12133](https://arxiv.org/abs/2305.12133) | not assessed | - | abstract |
| Ma, Wu, E 2020, *A Qualitative Study of the Dynamic Behavior for Adaptive Gradient Algorithms*, [2009.06125](https://arxiv.org/abs/2009.06125) | not assessed | - | abstract |
| Cattaneo, Shigida 2025, "default β2 may be too large" | id not found | - | not opened |

### Fixes on the gradient or the update (c03)
| Paper | Rule | Does v see the raw gradient? | Depth |
|---|---|---|---|
| Kumar et al. 2025, *ZClip*, [2504.02507](https://arxiv.org/abs/2504.02507) | z-score of the global norm, clip to μ + (z_thres²/z)σ | clipped (inference; not stated) | partial |
| Wang et al. 2026, *AdaGC*, [2502.11034](https://arxiv.org/abs/2502.11034) | per tensor, clip to 1.04 × EMA of its own clipped norm | clipped, into m and v | partial |
| Huang et al. 2026, *GradientStabilizer*, [2502.17055](https://arxiv.org/abs/2502.17055) | replace the norm by an EMA-based estimate, keep direction | modified, into m and v | partial |
| Wortsman et al. 2023, *StableAdamW*, [2304.13013](https://arxiv.org/abs/2304.13013) | divide the LR by max(1, RMS(g²/v)) per tensor | raw into both; only the LR changes | partial |
| Brock et al. 2021, *AGC*, [2102.06171](https://arxiv.org/abs/2102.06171) | per row, clip ‖G‖/‖W‖ at 0.01 | not confirmed | partial |
| Li et al. 2026, *CAST*, [2610.19761](https://arxiv.org/abs/2610.19761) | soft clip of singular values of the gradient standardised by the previous v | clipped, into m and v | partial |
| Huang et al. 2026, *Spectra*, [2602.11185](https://arxiv.org/abs/2602.11185) | replace the top-k singular directions of the momentum matrix by the tail level; no v | n/a | partial |
| Anonymous 2026, *Understanding and Mitigating AdamW Loss Spikes*, [OpenReview pAjrfibDB9](https://openreview.net/forum?id=pAjrfibDB9) | limit coordinates newly crossing \|m/√v\| > 1, plus a tensor-relative growth cap | unknown | abstract |
| Pascanu et al. 2013 [1211.5063](https://arxiv.org/abs/1211.5063); Zhang et al. 2019 [1905.11881](https://arxiv.org/abs/1905.11881); Koloskova et al. 2023 [2305.01588](https://arxiv.org/abs/2305.01588) | clipping theory | - | abstract |

### Fixes on the state: resets, restarts, rollback (c04, c11)
| Paper | Rule | Triggered by a spike? Selective? | Depth |
|---|---|---|---|
| Huang et al. 2025, *SPAM*, [2501.06842](https://arxiv.org/abs/2501.06842) | clip g_i to √(θ·v_i) when g_i² > θ·v_i (θ = 5000), clipped g into m and v; zero m and v every 500 steps, then 150 warmup steps | clip: per coordinate, at arrival. Reset: by the clock, everything | partial |
| Topollai, Choromanska 2026, [2603.16731](https://arxiv.org/abs/2603.16731) | periodic reset of v (and m) against quantisation staleness | no, by the clock | partial |
| Kim 2026, *Second-Moment Memory in Coordinatewise Adam*, [2608.15824](https://arxiv.org/abs/2608.15824) | a rare large gradient keeps the denominator large for about 1/(1−β2) steps | analysis only | partial |
| O'Donoghue, Candès 2012, *Adaptive Restart*, [1204.3982](https://arxiv.org/abs/1204.3982) | restart when the gradient opposes the momentum; wipes all momentum | all or nothing | partial |
| Liang et al. 2024, *Cautious Optimizers*, [2411.16085](https://arxiv.org/abs/2411.16085) | mask update coordinates whose sign disagrees with the gradient; m and v untouched | per coordinate, applied step only | partial |
| Cao, Li, Song 2025, *Grams*, [2412.17107](https://arxiv.org/abs/2412.17107) | sign from the current gradient, size from Adam | no | partial |
| Baydin et al. 2018, *Hypergradient Descent*, [1703.04782](https://arxiv.org/abs/1703.04782) | LR from ⟨new gradient, previous update⟩ | global LR only | partial |
| Chaubard et al. 2024, *Gradient Agreement Filtering*, [2412.18052](https://arxiv.org/abs/2412.18052) | drop micro-gradients that disagree, within one step | no memory across steps | partial |
| Shazeer, Stern 2018, *Adafactor*, [1804.04235](https://arxiv.org/abs/1804.04235) | clip the normalised update's RMS at 1 | output only | partial |
| Zhu et al. 2024, *Catapults in SGD*, [2306.04815](https://arxiv.org/abs/2306.04815) | SGD spikes live in the top few NTK directions and can help | analysis only | partial |
| Anonymous 2026, *Newer Is Not Always Better* (REBUILD_M), [OpenReview P0oFJZXAh8](https://openreview.net/forum?id=P0oFJZXAh8) | keep v, rebuild m from zero after checkpoint recovery; history-coverage bound | not about spikes | abstract |
| Asadi et al. 2023 [2306.17833](https://arxiv.org/abs/2306.17833); Ellis et al. 2024 [2412.17113](https://arxiv.org/abs/2412.17113); Ibrahim et al. 2024 [2403.08763](https://arxiv.org/abs/2403.08763); OpenReview seU1LYAiwm | resets in RL, inherited moments, state leakage | - | abstract |
| Biswas 2026 (OpenReview VJFfO4tLMB); Zhou et al. 2026, *The Hidden Ratio in Adam* (OpenReview GZha6judfp) | - | - | not opened |

### Redesigns of the second moment, ε and warmup (c05)
| Paper | Relevant point | Depth |
|---|---|---|
| Patitucci, Mokhtari 2026, *MVN-Grad*, [2602.10204](https://arxiv.org/abs/2602.10204) | Thm 3.3: per-update response to one spike is bounded independent of its size; Remark 3.1 bounds \|m/√v\|. No cumulative bound, nothing on repair | partial |
| Taniguchi et al. 2024, *ADOPT*, [2411.02853](https://arxiv.org/abs/2411.02853) | normalise by v_{t−1}, then momentum; clipped variant clips what enters m while v takes raw g² | partial |
| Reddi et al. 2018, *AMSGrad*, [1904.09237](https://arxiv.org/abs/1904.09237) | running max of v: a spike stays for good | partial |
| Zhou et al. 2019, *AdaShift*, [1810.00143](https://arxiv.org/abs/1810.00143) | v sees a gradient n steps late; the delay is fixed | partial |
| Liu et al. 2020, *RAdam*, [1908.03265](https://arxiv.org/abs/1908.03265) | rectifies the step while v has few samples; start of training only | partial |
| Kalra, Barkeshli 2024, *Why Warmup the LR?*, [2406.09405](https://arxiv.org/abs/2406.09405) | large early gradients keep v high and stall learning; treated only at initialisation | partial |
| Kosson et al. 2024, [2410.23922](https://arxiv.org/abs/2410.23922) | warmup need traced to m's bias correction and early update size | partial |
| Everett et al. 2024, [2407.05872](https://arxiv.org/abs/2407.05872) | ε must be scaled or it dominates small v | partial |
| Kassinos 2025, *Kourkoutas-β*, [2508.12996](https://arxiv.org/abs/2508.12996) | lowers β2 on a spike, per layer; m untouched | partial |
| Pagliardini et al. 2024, *AdEMAMix*, [2409.03137](https://arxiv.org/abs/2409.03137) | second, slow momentum (β3 = 0.9999); spike persistence in it not studied | partial |
| Zaheer et al. 2018, *Yogi* (NeurIPS) | rule not confirmed | abstract |

### Robust moment estimates (c07)
| Paper | Relevant point | Depth |
|---|---|---|
| Ilboudo et al. 2020, *TAdam*, [2003.00179](https://arxiv.org/abs/2003.00179) | Student-t weight on the gradient inside m only; v raw; weight fixed at arrival, per tensor | partial |
| Ilboudo et al. 2023, *AdaTerm*, [2201.06714](https://arxiv.org/abs/2201.06714) | one weight for m and the scale; fixed at arrival, per tensor | partial |
| Zhuang et al. 2020, *AdaBelief*, [2010.07468](https://arxiv.org/abs/2010.07468) | variance around m; a spike still inflates it | partial |
| Zhang et al. 2020, *ACClip*, [1912.03194](https://arxiv.org/abs/1912.03194) | clips the momentum with a threshold from raw \|g\|; no v | partial |
| Gorbunov et al. 2020 [2005.10785](https://arxiv.org/abs/2005.10785); Cutkosky, Mehta 2020, 2021 [2002.03305](https://arxiv.org/abs/2002.03305), [2106.14343](https://arxiv.org/abs/2106.14343) | clipping and normalised momentum under heavy tails; thresholds fixed in advance | partial |
| Yin et al. 2018 [1803.01498](https://arxiv.org/abs/1803.01498); Karimireddy et al. 2021 [2012.10333](https://arxiv.org/abs/2012.10333) | trimming across workers; centred clipping of momentum around the last aggregate | partial |
| Barndorff-Nielsen, Shephard 2004, bipower variation | jump-robust volatility | abstract |
| AdamMAD 2026, doi 10.1007/s41019-025-00341-7 | - | not opened |

### Filtering views (c08)
| Paper | Relevant point | Depth |
|---|---|---|
| Aitchison 2018, *AdaBayes*, [1807.07540](https://arxiv.org/abs/1807.07540) | Adam-like updates from a Kalman filter; Gaussian only, filtering only, no smoothing | full |
| Ollivier 2017, *Online Natural Gradient as a Kalman Filter*, [1703.00209](https://arxiv.org/abs/1703.00209) | equivalence of the two | partial |
| Duran-Martin et al. 2024, *WoLF*, [2405.05646](https://arxiv.org/abs/2405.05646) | down-weights an observation by its innovation size; weight set at arrival | partial |
| Adams, MacKay 2007, *Bayesian Online Changepoint Detection*, [0710.3742](https://arxiv.org/abs/0710.3742) | run-length posterior for regime changes | full |
| Chen, Liu 1993, JASA 88:284-297 | outlier types (additive, innovational, level shift, temporary change) | not opened; definitions are standard recall |

### Retroactive deletion, influence, state transplants (c09)
| Paper | Relevant point | Depth |
|---|---|---|
| Zhong et al. 2023, *Swift*, [2302.06173](https://arxiv.org/abs/2302.06173) | exact inverse of the LAST Adam/AdamW step, weights and m, v, for crash recovery | full |
| Graves et al. 2021, *Amnesiac ML*, [2010.10981](https://arxiv.org/abs/2010.10981) | store and subtract the weight updates of flagged batches; optimizer state not discussed | full |
| Wu et al. 2020, *DeltaGrad*, [2006.14755](https://arxiv.org/abs/2006.14755) | correct later gradients with an L-BFGS secant model; GD and SGD only, strong convexity | partial |
| Abdullah X 2025, *Unlearning at Scale*, [2508.12220](https://arxiv.org/abs/2508.12220) | exact replay from a log; m and v only through checkpoints | partial |
| Stewart 2026, [2605.17590](https://arxiv.org/abs/2605.17590) | misaligned-state view of unlearning, L-BFGS memory | full |
| Xu 2026, [2608.20442](https://arxiv.org/abs/2608.20442) | block transplants of w, m, v; m carries 89.9% of the signed effect, v about 0% | partial |
| Guo 2026, [2608.24593](https://arxiv.org/abs/2608.24593); Khanh 2026, [2607.06628](https://arxiv.org/abs/2607.06628); Li 2026, [2607.07206](https://arxiv.org/abs/2607.07206) | block-level state interventions; none retroactive, none per coordinate | partial / full |
| Bae et al. 2024, *SOURCE*, [2405.12186](https://arxiv.org/abs/2405.12186); Wang et al. 2024, *In-Run Data Shapley*, [2406.11011](https://arxiv.org/abs/2406.11011) | attribution from gradient inner products; SGD only; used to filter and retrain, never to act in the run | partial |
| Ren et al. 2018, *Learning to Reweight Examples*, [1803.09050](https://arxiv.org/abs/1803.09050) | weight = clipped alignment with a clean validation gradient; about 3× cost; weight never revised | full |
| Pruthi et al. 2020, *TracIn*, [2002.08484](https://arxiv.org/abs/2002.08484) | - | abstract |
| OLMo Team 2025, *2 OLMo 2 Furious*, [2501.00656](https://arxiv.org/abs/2501.00656) | spike score at 7σ; a skip-step rule was **not found** in the text read | partial |
| Chowdhery et al. 2022, *PaLM*, [2204.02311](https://arxiv.org/abs/2204.02311) | restart about 100 steps back, skip 200 to 500 batches; cause is "batch × parameter state" | partial (Sec. 5.1) |

### Not re-read in this run
- Stability theory and spikes at scale (clusters c06, c02): Cohen et al. 2022 [2207.14484](https://arxiv.org/abs/2207.14484), central
  flows [2410.24206](https://arxiv.org/abs/2410.24206), Liu and Li 2026 [2608.19762](https://arxiv.org/abs/2608.19762), Wortsman et al. 2023
  [2309.14322](https://arxiv.org/abs/2309.14322), Takase et al. 2023 [2312.16903](https://arxiv.org/abs/2312.16903). These are in the project's earlier
  corpus (`PLAN_V8.md` §5, `references.bib`). This run relies on that reading and adds nothing to it.
- Control, numerics and statistics classics (cluster c10: anti-windup, recursive least squares with forgetting,
  trust-region ratios, influence functions, sequential tests) are background, cited from memory and **unverified**.

## 2. Gaps

Type: **stated** (the authors say so), **hidden** (nobody mentions it), **undefined** (no formal statement exists),
**conflict** (sources disagree or a common belief is unsupported).

| # | Gap | Evidence | Type | What an optimal repair needs |
|---|---|---|---|---|
| G1 | Nothing edits m and v selectively after a detected spike. State-side fixes are full or clock-driven resets. | SPAM, Topollai, Asadi, Ellis; Swift inverts only the last step; reader search found none (queries in `c09`, question 11) | hidden | a receipt for one past step that can be removed later, on its own support |
| G2 | Every weight or clip is fixed when the gradient arrives. No rule revises it from later gradients. | TAdam Eq. 9, AdaTerm Eq. 13, ACClip, WoLF, ADOPT's one-step lag, AdaShift's fixed delay, Learning to Reweight | hidden | a decision that waits for evidence, which needs an exact undo |
| G3 | m and v are never treated as two channels with a stated trade-off. Clippers feed the same modified gradient to both; ADOPT-clip, TAdam and ACClip limit m only, with no argument about the after-effect. | c03 question 3; c07 gap 1 | undefined | a law for total displacement as a function of what each channel keeps |
| G4 | Bounds are per update. Nobody gives the total displacement one large gradient causes. | MVN-Grad Thm 3.3 and Remark 3.1; Kim 2026 (β1 = 0, d = 1); ADOPT | undefined | the cumulative echo and where it saturates |
| G5 | Nobody measures the lasting effect of a spike against a no-spike run on the same data order. Results are final loss, spike counts, or one-step loss. | c04 question 5; SPAM; pAjrfibDB9 (abstract); PaLM's evidence is one restart | hidden | exact forks, and a bound on what can last |
| G6 | No optimizer-side method tells a one-off outlier from a real change or from an instability. | every entry: "outlier vs regime: no"; Bai et al. explain instability but only prevent it | hidden | one test that separates the three and prices each action |
| G7 | Agreement rules are all-or-nothing or act on the output only. Restart wipes all momentum; Cautious masks the step and leaves m poisoned. | O'Donoghue and Candès Sec. 3.2; Cautious Alg. 2; Cautious lists eigenspace masking as future work (Sec. 5) | stated + hidden | cancel only the flagged event's part of m |
| G8 | Retroactive correction exists only for weights and by replay. The secant correction of later gradients is proved for GD/SGD under strong convexity. | Amnesiac, DeltaGrad, 2508.12220 | stated (DeltaGrad's scope) | a return of the weights that works with Adam and needs no replay |
| G9 | Instability control is preventive and looks at update size. An outlier's echo sits at m/√v ≤ 0.447, inside the normal range, so those rules cannot see it. | Bai et al.; pAjrfibDB9 (abstract); StableAdamW; Adafactor; the invisibility is our inference | hidden | curvature measured along the event, and a v level that is derived |
| G10 | Nobody shrinks an inflated v. AMSGrad makes it permanent; GI-Adam and RAdam treat v only at the start; the stall is described at initialisation only. | c05 gaps 4 and 6; Kalra and Barkeshli Sec. 5 | hidden | remove v's inflation, but only together with m's share |
| G11 | No look at optimizer state after an injected spike at LLM scale. Long memories (β2 = 0.999, AdEMAMix β3 = 0.9999) are unmeasured. | c05 gap 7; c07 gap 7; MVN-Grad Sec. 5 lists larger scale as future work | stated | a regime map: when repair matters |
| G12 | Influence scores exist per step but are SGD-only and never used to act in the run; the method that acts needs clean data and 3× cost. | In-Run Data Shapley Sec. 6 (Adam is future work); Learning to Reweight Sec. 3.3 | stated | the same inner products, under Adam's preconditioner, from the training stream itself |
| G13 | SPAM reports no numeric ablation that separates its clip from its reset. | reader B, question 1 | conflict | a prediction for which part does the work |
| G14 | The skip-step rule usually credited to OLMo 2 was not found in the OLMo 2 text that was read; the 7σ rule there is a spike *score*. | reader C, question 10; marked uncertain | conflict | check the source before citing skip-step as OLMo's |

## 3. Future work the authors name (quoted locations)
- Cautious Optimizers, Sec. 5: masking in eigenspace rather than parameter space; multi-step analysis is "challenging" (Sec. 2.3).
- MVN-Grad, Sec. 5: heavy-tailed noise and larger-scale pretraining; no universal robustness claimed (Sec. 3.2).
- In-Run Data Shapley, Sec. 6: extension to Adam.
- TAdam, Sec. V: adapting ν to the presence of outliers.
- AdaShift, Sec. 4.2: "We leave further investigations on φ as future work."
- Kalra and Barkeshli, Sec. 7: dependence on β1, β2, ε unexplored; small scale only.
- Adaptive restart, Sec. 4.6: behaviour away from the optimum "unclear"; no stochastic gradients.
- DeltaGrad: bounds need strong convexity; momentum and Adam not covered.
- PaLM, Sec. 5.1: no principled mitigation for spikes.

## 4. Open reading debts
1. Three ICLR 2027 submissions are known only from abstracts: pAjrfibDB9 (the closest mitigation), P0oFJZXAh8
   (REBUILD_M and history coverage), seU1LYAiwm. Their full texts must be read in a browser before submission.
2. Biswas 2026 and *The Hidden Ratio in Adam* were not opened. Both concern v's history, so they may bear on G4 and G10.
3. Yogi, AdamMAD, Chen and Liu 1993 and TracIn are cited at abstract level or from memory.
4. G14 needs one careful read of OLMo 2, Sections 4 and 5.

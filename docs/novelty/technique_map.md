# Technique map and combination matrix

*Written 2026-11-12. Companion to `rml3_method.md`. This file is the wide, shallow survey (Steps 1 and 2 of the
brief); the three chosen approaches are specified in `rml3_method.md`. Nothing was run.*

**How to read the "Check" column.** `id` = the arXiv id and title were confirmed today against the arXiv listing;
the paper was **not** re-read, so its content counts as SKIMMED. `abs` = an abstract summary was read today.
`snip` = seen only as a search snippet. `ledger` = already in `gap_ledger.md` at the depth recorded there.
`W` = an encyclopedia page, not opened today (UNVERIFIED link). Mechanisms are stated from general knowledge.

**How to read "Status".** `ALREADY DONE` = something already applies this family to optimizers, optimizer state or
spikes; the paper is named. `IN V8` = the idea is already in `PLAN_V8.md` Appendix A under that id.
`not found` = one search today found no application to optimizer state or spikes; that bounds nothing beyond the
one search.

The brief names "V6". `PLAN_V8.md` is the governing plan and keeps V6's four components: **C1** the fork-and-compare
instrument, **C2** the localizer, **C3** the repair, and the theory. Everything below maps to those.

---

## 1. The map (75 entries)

### A. Teacher–student and distillation
| # | Technique | Built for | Mechanism | Source | Use for optimizer state or spikes | Status | Check |
|---|---|---|---|---|---|---|---|
| 1 | Knowledge distillation | compressing a model | student matches the teacher's soft outputs | [1503.02531](https://arxiv.org/abs/1503.02531) | pull the post-spike model's outputs back toward a pre-spike snapshot on fresh data: a rollback in function space | not found for spikes | id |
| 2 | Mean Teacher | semi-supervised learning | teacher = running average of student weights; consistency loss | [1703.01780](https://arxiv.org/abs/1703.01780) | averaged weights as a free restore point and as the judge in a remedy tournament | used to stabilise self-distillation ([2601.19897](https://arxiv.org/abs/2601.19897), snip), not for spikes | id |
| 3 | BYOL | self-supervised vision | online net chases a slowly averaged target net | [2006.07733](https://arxiv.org/abs/2006.07733) | a slow "target" copy of m and v as the reference state | ALREADY DONE in part: AdEMAMix's slow momentum (ledger) | id |
| 4 | DINO | self-supervised vision | self-distillation with centering | [2104.14294](https://arxiv.org/abs/2104.14294) | centering for per-example loss scores so drift does not raise alarms | weak fit | id |
| 5 | Born-again networks | accuracy by re-training | a same-size student learns from the teacher | [1805.04770](https://arxiv.org/abs/1805.04770) | replay the spike window with the pre-spike model as teacher | weak fit | id |
| 6 | Policy distillation | compressing a policy | regress a small policy on a large one | [1511.06295](https://arxiv.org/abs/1511.06295) | distil the fork oracle (the clean-twin correction) into a cheap rule | IN V8 (R8) | id |

### B. Prediction and forecasting
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 7 | Learned optimizers | replacing hand-made update rules | a network outputs the update | [1606.04474](https://arxiv.org/abs/1606.04474), [2211.09760](https://arxiv.org/abs/2211.09760) | a learned verdict or repair policy | ALREADY DONE as optimizers; IN V8 (R8) | id |
| 8 | Learning-curve extrapolation | stopping bad runs early | predict the rest of a curve from its start | [2310.20447](https://arxiv.org/abs/2310.20447) | predict the no-spike loss and the recovery time without a twin | not found for spikes | id |
| 9 | Hyperband | hyperparameter search | stop the worst partial runs, keep the rest | [1603.06560](https://arxiv.org/abs/1603.06560) | stop losing remedy rollouts early | not found | id |
| 10 | Population Based Training | hyperparameter schedules | copy the better member, perturb, continue | [1711.09846](https://arxiv.org/abs/1711.09846) | a tournament of remedies after a spike | not found for spikes (2 searches) | id |
| 11 | Short-horizon bias | meta-optimization theory | choices judged on short rollouts favour greedy options | [1803.02021](https://arxiv.org/abs/1803.02021) | the main threat to any rollout-based choice, and measurable with exact twins | not found for remedies | id |
| 12 | Hazard and survival models | time-to-event data | model the event rate over time | [W](https://en.wikipedia.org/wiki/Kaplan%E2%80%93Meier_estimator) | time to recovery; spike risk | ALREADY DONE: TIC forecasts spikes (ledger); IN V8 (#25) | W |
| 13 | Early-warning signals | tipping points in nature | autocorrelation and variance rise before a transition | [doi](https://doi.org/10.1038/nature08227), [doi](https://doi.org/10.1073/pnas.1424320112) | warning for spikes that build up | not found for training (1 search) | abs |
| 14 | Short differentiable rollouts | choosing training schedules | differentiate 8 AdamW steps through model and optimizer state | [2608.24593](https://arxiv.org/abs/2608.24593) | choose an intervention by a short look ahead | ALREADY DONE for loss schedules | abs |

### C. Anomaly and change-point detection
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 15 | CUSUM | quality control | cumulative sum of deviations crosses a line | [W](https://en.wikipedia.org/wiki/CUSUM) | lead-time detector | IN V8 (#61) | W |
| 16 | Bayesian online change-point detection | regime shifts | posterior over the time since the last change | [0710.3742](https://arxiv.org/abs/0710.3742) | tell a one-off outlier from a lasting shift | ledger | ledger |
| 17 | Isolation forest | tabular outliers | anomalies are isolated by few random splits | [W](https://en.wikipedia.org/wiki/Isolation_forest) | cheap detector on per-tensor statistics | not found | W |
| 18 | Autoencoder reconstruction error | outliers in signals | a model of "normal" reconstructs anomalies badly | [W](https://en.wikipedia.org/wiki/Autoencoder) | a learned picture of normal optimizer statistics | not found | W |
| 19 | Conformal prediction | calibrated uncertainty | thresholds from ranks of recent scores | [2107.07511](https://arxiv.org/abs/2107.07511) | a stated false-alarm rate for flags and verdicts | IN V8 (#60, R32) | id |
| 20 | Robust z-score rules in practice | production LLM runs | flag at a local z-score; treat narrow and wide spikes differently | [2512.06201](https://arxiv.org/abs/2512.06201) | rule-based triage | ALREADY DONE (K2-V2) | snip |

### D. Model and parameter editing
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 21 | ROME | editing one fact | a rank-one weight edit at a causally traced site | [2202.05262](https://arxiv.org/abs/2202.05262) | a rank-one state edit at the traced damage site | IN V8 (S11, framing) | id |
| 22 | MEMIT | editing many facts | many edits solved jointly | [2210.07229](https://arxiv.org/abs/2210.07229) | undo several spikes at once | not found | id |
| 23 | Task vectors | adding or removing a skill | θ_finetuned − θ_base; negate to forget | [2212.04089](https://arxiv.org/abs/2212.04089) | a "damage vector" = spiked state − clean-twin state, which can be scaled and subtracted | not found on optimizer state (1 search) | id |
| 24 | Model soups | accuracy | average several fine-tuned models | [2203.05482](https://arxiv.org/abs/2203.05482) | average the top remedy arms | not found | id |
| 25 | WiSE-FT | robust fine-tuning | interpolate two models | [2109.01903](https://arxiv.org/abs/2109.01903) | interpolate between "keep" and "undo" | IN V8 (W11) | id |
| 26 | Fisher merging | merging models | weight each parameter by its Fisher value | [2111.09832](https://arxiv.org/abs/2111.09832) | Adam's v as the curvature weight | ALREADY DONE: OTA merging ([2509.11167](https://arxiv.org/abs/2509.11167), abs); "Fishers for Free" ([2507.18807](https://arxiv.org/abs/2507.18807), id) | id |
| 27 | TIES merging | merging models | drop small entries, settle sign conflicts | [2306.01708](https://arxiv.org/abs/2306.01708) | drop the spike's share where it fights the honest momentum | close to Cautious (ledger) | id |
| 28 | Latest weight averaging | faster training | average recent checkpoints | [2209.14981](https://arxiv.org/abs/2209.14981) | averaged recent weights as a restore point | not found for spikes | id |

### E. Continual learning and forgetting
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 29 | EWC | not forgetting old tasks | a Fisher-weighted pull to old weights | [1612.00796](https://arxiv.org/abs/1612.00796) | anchor the recovery, with pre-spike v as the Fisher | cut by the council (resists normal progress) | id |
| 30 | GEM | continual learning | project gradients so stored-task loss does not rise | [1706.08840](https://arxiv.org/abs/1706.08840) | project the spike direction out of later updates | IN V8 (#44) | id |
| 31 | Orthogonal gradient descent | continual learning | orthogonal projection against old gradients | [1910.07104](https://arxiv.org/abs/1910.07104) | same | IN V8 (#44) | id |
| 32 | SISA unlearning | cheap deletion | shard data so a deletion retrains little | [1912.03817](https://arxiv.org/abs/1912.03817) | micro-batch shards make exact removal cheap | IN V8 (R14) | id |
| 33 | Certified removal | provable deletion | a Newton step removes a point's influence | [1911.03030](https://arxiv.org/abs/1911.03030) | a one-step weight correction | not found for Adam | id |
| 34 | DeltaGrad | fast re-training after deletion | correct later gradients with a secant model | [2006.14755](https://arxiv.org/abs/2006.14755) | return the weights without replay | ledger gap G8: Adam not covered | ledger |
| 35 | Amnesiac ML | deletion | subtract the logged updates | [2010.10981](https://arxiv.org/abs/2010.10981) | open-loop removal of the spike's weight footprint | ALREADY DONE for weights | ledger |
| 36 | Second moments across tasks | continual learning | keep a weighted average of v from earlier tasks | [2404.02754](https://arxiv.org/abs/2404.02754) | a reference v that outlives an event | ALREADY DONE (tasks, not spikes) | abs |

### F. Robust statistics and optimization
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 37 | Trimmed means, median of means | heavy-tailed data | discard or down-weight the extremes | [1906.04280](https://arxiv.org/abs/1906.04280) | trim outlier examples before they reach m and v | ALREADY DONE inside moments: TAdam, AdaTerm (ledger) | id |
| 38 | Small-loss selection (Co-teaching) | noisy labels | train on the low-loss examples | [1804.06872](https://arxiv.org/abs/1804.06872) | drop high-loss examples | ALREADY DONE; loss-based reweighting for LLM pretraining ([2502.06733](https://arxiv.org/abs/2502.06733), abs) | id |
| 39 | Late re-selection (Early Cutting) | noisy labels | a later model re-judges examples picked early | [2502.08227](https://arxiv.org/abs/2502.08227) | precedent for revising an early data decision | multi-epoch, not single-pass | snip |
| 40 | Stochastic weight averaging | generalisation | average weights along the path | [1803.05407](https://arxiv.org/abs/1803.05407) | averaged weights as judge or restore point | not found for spikes | id |
| 41 | Lookahead | stable optimization | slow weights follow fast weights every k steps | [1907.08610](https://arxiv.org/abs/1907.08610) | slow weights are a free rollback point | not found as a spike remedy | id |
| 42 | SAM | flat minima | step against the worst nearby perturbation | [2010.01412](https://arxiv.org/abs/2010.01412) | probe sharpness on the hit support | not pursued | id |
| 43 | Trust regions | safe policy updates | accept a step only if real and predicted gains agree | [1502.05477](https://arxiv.org/abs/1502.05477) | step rejection is skip-step with a test | ledger (StableAdamW is a relative) | id |
| 44 | Gradient agreement filtering | noisy micro-batches | drop micro-gradients that disagree | [2412.18052](https://arxiv.org/abs/2412.18052) | example-level filtering | ALREADY DONE, irreversible | ledger |
| 45 | Asynchronous SGD | parallel training | apply gradients that are a few steps stale | [1506.08272](https://arxiv.org/abs/1506.08272) | licence for adding an old gradient back late | SGD only | abs |

### G. Meta-learning and hypernetworks
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 46 | MAML | fast adaptation | learn a start that adapts in a few steps | [1703.03400](https://arxiv.org/abs/1703.03400) | meta-learn a repair | IN V8 (R8) | id |
| 47 | HyperNetworks | generating weights | one net outputs another's parameters | [1609.09106](https://arxiv.org/abs/1609.09106) | predict per-tensor memory length after a spike | ALREADY DONE in spirit: Kourkoutas-β (ledger) | id |
| 48 | Metagradient descent | tuning training itself | exact gradients through the whole run | [2503.13751](https://arxiv.org/abs/2503.13751) | gradient of final loss with respect to m and v | ALREADY DONE in part: Xu ([2608.20442](https://arxiv.org/abs/2608.20442)), Guo ([2608.24593](https://arxiv.org/abs/2608.24593)) | abs |
| 49 | Learning to Reweight Examples | noisy and biased data | weight = clipped alignment with a clean batch's gradient | [1803.09050](https://arxiv.org/abs/1803.09050) | the verdict statistic, with later training batches as the clean set | ledger gap G12 | ledger |

### H. Interpretability
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 50 | Activation patching | finding circuits | swap an activation from another run | [2404.15255](https://arxiv.org/abs/2404.15255) | swap m or v from the clean twin | ALREADY DONE: Xu 2026; IN V8 | abs |
| 51 | Path patching | circuit wiring | patch along one path only | [2211.00593](https://arxiv.org/abs/2211.00593) | which tensor's poison reaches which | IN V8 (R26) | id |
| 52 | Attribution patching (AtP\*) | fast localisation | a gradient estimate of every patch | [2403.00745](https://arxiv.org/abs/2403.00745) | first-order score for every coordinate | IN V8 (#11) | abs |
| 53 | Linear probes | reading representations | a linear readout of a property | [1610.01644](https://arxiv.org/abs/1610.01644) | read "a spike happened k steps ago" from the state | IN V8 (#41) | id |
| 54 | Sparse autoencoders | finding features | an overcomplete sparse dictionary | [2309.08600](https://arxiv.org/abs/2309.08600) | a dictionary of damage vectors | not found on optimizer state (1 search); IN V8 (R4) | id |
| 55 | Influence, TRAK, Data Shapley | blaming training data | gradient inner products with a target | [1703.04730](https://arxiv.org/abs/1703.04730), [2303.14186](https://arxiv.org/abs/2303.14186), [2406.11011](https://arxiv.org/abs/2406.11011) | which examples wrote the damage | ALREADY DONE for data selection: GREATS (snip), OPUS ([2602.05400](https://arxiv.org/abs/2602.05400), snip), Filter-then-Weight ([2604.00001](https://arxiv.org/abs/2604.00001), abs) | id |

### I. Compression and low-rank methods
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 56 | GaLore | memory-light training | run the optimizer in a low-rank gradient subspace | [2403.03507](https://arxiv.org/abs/2403.03507) | repair only in the spike's low-rank subspace | not found for repair | id |
| 57 | LoRA | cheap fine-tuning | low-rank weight updates | [2106.09685](https://arxiv.org/abs/2106.09685) | store the receipt as low-rank factors | ALREADY DONE for optimizer state: LoRA-Pre ([2602.24283](https://arxiv.org/abs/2602.24283), abs) | id |
| 58 | SM3 | memory-light Adagrad | second moments shared over cover sets | [1901.11150](https://arxiv.org/abs/1901.11150) | a cheap reference v | not found for repair | id |
| 59 | Count-sketch optimizers | memory-light Adam | hash the state into a sketch | [1902.00179](https://arxiv.org/abs/1902.00179) | sketched receipts | IN V8 (X3, R29) | id |
| 60 | Ghost norms | private training | per-example gradient norms without per-example gradients | [1510.01799](https://arxiv.org/abs/1510.01799), [2110.05679](https://arxiv.org/abs/2110.05679), [2210.00038](https://arxiv.org/abs/2210.00038) | flag outlier examples cheaply; store their share as per-token factors | per-example statistics in optimizers are studied by Roulet et al. ([2510.00236](https://arxiv.org/abs/2510.00236), abs); not for spikes | id |

### J. Control and estimation
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 61 | Kalman filter | tracking | blend prediction and measurement by their variances | [W](https://en.wikipedia.org/wiki/Kalman_filter) | Adam as a filter; x as a normalised innovation | ALREADY DONE: AdaBayes, Ollivier (ledger) | ledger |
| 62 | Fixed-lag smoothing with gating | navigation | revise an estimate a few steps later; reject gated outliers | [2109.14220](https://arxiv.org/abs/2109.14220) | the probation verdict | not found for optimizers (1 search) | snip |
| 63 | PID control | process control | act on error, its sum and its change | [doi](https://doi.org/10.1109/CVPR.2018.00889) | momentum as an integrator | ALREADY DONE (An et al., CVPR 2018) | snip |
| 64 | Anti-windup | saturating actuators | stop the integrator when the output saturates | [W](https://en.wikipedia.org/wiki/Integral_windup) | the echo is integrator windup | IN V8 (#22, K7) | W |
| 65 | Model-predictive control | constrained plants | simulate candidate actions, apply the best, repeat | [W](https://en.wikipedia.org/wiki/Model_predictive_control) | the tournament: the deterministic run is its own exact model | not found | W |
| 66 | Integral quadratic constraints | robust control | certify stability of a linear system with a bounded nonlinearity | [1408.3595](https://arxiv.org/abs/1408.3595) | a self-healing proof | not pursued | abs |

### K. Imports from other fields
| # | Technique | Built for | Mechanism | Source | Use | Status | Check |
|---|---|---|---|---|---|---|---|
| 67 | Mediation with interaction | epidemiology | split an effect into four parts | [doi](https://doi.org/10.1097/EDE.0000000000000121) | the 2 × 2 reading in `upgraded_plan.md` §2 | IN V8 (T2) | abs |
| 68 | Synthetic control | policy evaluation | build the missing control from donors | [W](https://en.wikipedia.org/wiki/Synthetic_control_method) | a twin-free counterfactual for natural spikes | IN V8 (#39); not found on training curves | W |
| 69 | Interrupted time series | public health | forecast from the unit's own past | [W](https://en.wikipedia.org/wiki/Interrupted_time_series) | same, with no donors | not found | W |
| 70 | Prediction-powered inference | science with ML labels | many cheap predictions, debiased by a few gold labels, give valid intervals | [2301.09633](https://arxiv.org/abs/2301.09633) | cheap first-order patch predictions + a few exact forks | not found with patching (2 searches) | abs |
| 71 | Write-ahead logging | databases | log enough to undo any step | [W](https://en.wikipedia.org/wiki/Algorithms_for_Recovery_and_Isolation_Exploiting_Semantics) | the Reversible Moment Log | IN V8 (#5) | W |
| 72 | Speculative execution | processors | run ahead on a guess; keep or squash | [W](https://en.wikipedia.org/wiki/Speculative_execution) | snapshot, continue, then commit or squash | not found | W |
| 73 | Group testing and fine-mapping | screening, genetics | test pools; assign credit among correlated causes | [W](https://en.wikipedia.org/wiki/Group_testing) | the localizer's oracle | IN V8 (#6) | W |
| 74 | Fault tolerance (checks, redundancy) | reliable computing | detect and mask silent errors | [2502.12340](https://arxiv.org/abs/2502.12340) | faults in m and v | IN V8 (R22, #15) | abs |
| 75 | Skip and re-inject, in practice | a 300B-parameter run | skip the spiking update, feed its data again later | [2503.05139](https://arxiv.org/abs/2503.05139) | a data-level second chance | ALREADY DONE; details UNVERIFIED (the PDF could not be read) | snip |

---

## 2. Holes in the closest papers

| Paper | What it leaves open | Evidence | Which component uses the hole |
|---|---|---|---|
| GREATS, OPUS, Filter-then-Weight | The score needs a validation or target set, and the choice is final when the batch arrives | snip / abs | Appeal: evidence from later training batches, and a decision that can be reversed |
| Data Shapley in One Training Run | Adam is left to future work; scores are used after the run | ledger (its Sec. 6) | Appeal's verdict under Adam's scaling, used inside the run |
| Learning to Reweight Examples | Needs a clean set and about three times the cost | ledger | same |
| Ling (skip and re-inject) | The second chance is unconditional; nothing tests whether the data was harmful | snip, UNVERIFIED | Appeal's evidence test |
| K2-V2 | Narrow and wide spikes are told apart by a rule; nothing checks the rollback was needed | snip, UNVERIFIED | the tournament |
| PaLM | "No principled mitigation" | ledger (its Sec. 5.1) | the tournament |
| SPAM | No number separates its clip from its reset | ledger G13 | prereg-v3 P5 |
| Guo 2026 | Eight-step horizons, loss schedules only | abs | the tournament (long horizons, remedies, exact replay); the costate ceiling |
| Or 2026 ([2601.17483](https://arxiv.org/abs/2601.17483)) | Evaluates only accept and rollback; richer strategies are named as future work | page (added 2026-11-13) | the tournament's menu; also a new baseline |
| AutoLRS ([2105.10762](https://arxiv.org/abs/2105.10762)) | Short trials with extrapolation, but on a schedule and for learning rates only | abs (added 2026-11-13) | the tournament's extrapolated judge |
| Jiang et al. 2026 ([2608.04408](https://arxiv.org/abs/2608.04408)) | Branch outcomes steer training, but for a student's reasoning steps, not optimizer state | abs (added 2026-11-13) | the remedy policy (build plan §10) |
| Xu 2026 | The costate predicts signs; no spikes, no best edit | abs | the costate ceiling |
| Metagradient descent | Needs "smooth" training; does not say where training is smooth | abs | the measured chaos floor says where a costate can be trusted |
| Short-horizon bias (Wu et al.) | The bias is shown for learning rates, on short rollouts | id only | measured for remedies against end-of-run truth |
| Heimersheim & Nanda; AtP\* | Patching effects come without intervals; AtP\* bounds missed components, not effect sizes | abs | prediction-powered patching |
| DeltaGrad | Proved for SGD under strong convexity; momentum and Adam not covered | ledger | weight restitution (C7 below) |
| Kwok et al. | Shows sensitivity; gives no unit for judging interventions | abs | the floor as a ruler (V8) |

---

## 3. The fifteen entries carried forward

E1 ghost norms (60) · E2 gradient-alignment scores (49, 55) · E3 the skip identity and undo log (71) ·
E4 conformal thresholds (19) · E5 slow teacher state (2, 3) · E6 damage vectors (23) · E7 tournaments and
speculation (10, 65, 72) · E8 costate (48) · E9 prediction-powered inference (70) · E10 synthetic control (68, 69) ·
E11 secant correction (34) · E12 low-rank factors (56, 57) · E13 fixed-lag smoothing (62) · E14 forecasts (12, 13) ·
E15 sparse dictionaries (54).

## 4. Combination matrix

Scores are 1 to 5. **N** novelty after today's searches, **P** plausibility, **C** cost (5 = cheap), **S** how much
it strengthens the paper's core claim. They are my judgment, not measurements.

| ID | Pieces | One-sentence idea | Extends | Why the pieces fit | Why it might fail | N | P | C | S |
|---|---|---|---|---|---|---|---|---|---|
| **K1 Appeal** | E1 + E2 + E3 (+E4) | Withhold flagged examples' gradient share, then add it back into m and v exactly if later batches agree with it | C3; detector | The skip identity makes late admission exact; alignment scores supply the evidence; ghost factors make the record small | Later batches cannot confirm rare but real examples; the stale gradient may no longer fit | 4 | 4 | 4 | 4 |
| **K2 Tournament** | E7 + E3 + E4 | After a spike, replay a short menu of remedies from a snapshot on the same data and keep the best; "do nothing" is always on the menu | C1 becomes a controller; C3 | Bitwise replay is an exact simulator; every remedy becomes oracle-timed; harm is bounded at the horizon | Short horizons mis-rank (entry 11); snapshot memory at scale | 3 | 5 | 3 | 5 |
| **K3 Ceiling** | E8 + E9 | The costate bounds what any state edit can gain; its cheap predictions plus a few exact forks give valid intervals | C2; theory | The harness supplies exact labels; the costate supplies thousands of predictions | The linearisation breaks at high dose (the interaction in `upgraded_plan.md` §2); engineering | 3 | 3 | 2 | 4 |
| K4 Teacher restore | E5 + E3 + anti-windup | With no receipt, restore m and v on flagged coordinates from slow copies, m first and v second | C3 | Slow copies barely feel one outlier | Twice the optimizer memory; close to V8's band clamp (W3) and R7 | 2 | 4 | 3 | 3 |
| K5 Damage dictionary | E6 + E15 + E12 | Learn a sparse or low-rank basis of damage vectors from many exact twins, then repair twin-free by projecting onto it | C2, C3 | Twins are cheap at proxy scale | Damage may not share a basis across runs; overlaps V8 R4, R8, R16 | 3 | 2 | 2 | 4 |
| K6 Twin-free damage | E10 + E14 | Estimate the missing clean run from the run's own past and from donors, scored against real twins | C1 | The harness can grade any estimator | Training curves drift; overlaps V8 #39 | 2 | 3 | 5 | 3 |
| K7 Weight restitution | E11 + E3 | Subtract the spike's logged weight footprint, shrunk by an attenuation curve measured on exact twins | C3 | The footprint is exact in open loop; twins give the closed-loop factor | At realistic dose the direct write is already tiny | 3 | 3 | 4 | 2 |
| K8 Smoothed verdict | E13 + E2 | Treat the verdict as a fixed-lag posterior between "outlier" and "real change" | C3 (RML-2) | Same statistic, principled stopping | Model mismatch; small supports | 3 | 3 | 5 | 2 |
| K9 Low-rank receipts | E12 + E3 | Store each receipt as its top singular factors and measure damage removed against rank | C3 | A bad batch's gradient is a sum of few outer products | v's share is an elementwise square and is not low-rank; LoRA-Pre is close | 2 | 4 | 4 | 3 |
| K10 Pruned tournament | E7 + E8 | Use costate predictions to shortlist the menu | C1 | Saves rollouts | Guo 2026 already shortlists by derivatives | 1 | 4 | 3 | 2 |
| K11 Armed snapshots | E14 + E7 | A forecast decides when to keep the snapshot the tournament needs | C1 | Cuts snapshot cost | Bad-batch spikes have no precursor | 2 | 3 | 4 | 2 |
| K12 Per-example v | E1 + robust means | Build v from per-example squares so one bad example inflates it more | theory | Stronger built-in protection is predicted | Roulet et al. already study per-example preconditioners | 1 | 3 | 3 | 2 |
| K13 Calibrated appeal | E4 + E2 | Set the verdict threshold from the same statistic on ordinary examples | C3 | Removes a hand-set threshold | Needs shadow records | 3 | 4 | 4 | 3 |
| K14 Scaled-up estimates | E9 + E10 | A few exact twins debias many twin-free estimates of natural-spike damage | C1 at scale | The right tool when twins are scarce | Needs natural spikes with saved state | 3 | 3 | 2 | 4 |
| K15 Soup of remedies | E7 + model soups | Average the best two arms instead of picking one | C3 | Hedges a close call | Averages of states may be off the reachable set | 2 | 2 | 4 | 1 |

**Chosen:** K1, K2 and K3. K13 is folded into K1; K9 is K1's storage format; K11 is how the forecast layer of RML-2
feeds K2; K4 stays as the fallback V8 already plans (the band clamp). K5, K6, K7 and K14 are kept as later options.

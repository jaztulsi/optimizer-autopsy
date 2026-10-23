# 09 · Detailed quantitative analysis (2026-10-22)

> **v2 correction (2026-10-22):** §1.4(3) holds open loop only. Closed loop, the underdamped (w, m) modes decay at √β1 per step, leaving ≈ 7% at s1+50, not ≈ 0.5% (`10_wave2_scale_theory.md` T3″). The 5.84η / 1.165 / k½ ≈ 158 numbers were re-derived and hold.

*Derivations done by hand from the Adam recurrences and the repo's measured timings. **Nothing was executed.**
Every number marked [I] is my own arithmetic; please re-check it. [V] means checked in the repo or a cited
source. AdamW: m_t = β1·m_{t−1} + (1−β1)·g_t and v_t = β2·v_{t−1} + (1−β2)·g_t², where g_t is the
**post-clip** gradient. Weight decay acts on w only. The project default is (β1, β2) = (0.9, 0.95) [V:
proxy config].*

---

## 1 · How long Adam remembers a spike

### 1.1 Memory constants

| Quantity | β = 0.9 (m) | β2 = 0.95 | β2 = 0.99 | β2 = 0.999 |
|---|---|---|---|---|
| −ln β | 0.1054 | 0.05129 | 0.01005 | 0.0010005 |
| e-fold time 1/(−ln β) (steps) | 9.5 | 19.5 | 99.5 | 999.5 |
| half-life ln2/(−ln β) (steps) | 6.6 | 13.5 | 69.0 | 692.8 |
| share of a step-s contribution still present after 100 steps | 2.7e-5 | 5.9e-3 | 0.366 | 0.905 |

[I] These are pure algebra on β^k.

### 1.2 Planted inflation (`v_inflate`: v ← κ·v at one step)
With v ≈ ḡ² beforehand, v_k/ḡ² = 1 + (κ−1)·β2^k. Two consequences:
- **Back within 2× of normal** when k ≥ ln(κ−1)/(−ln β2):

| κ | β2 = 0.95 | β2 = 0.99 | β2 = 0.999 |
|---|---|---|---|
| 10² | 89.6 | 457 | 4,593 |
| 10⁴ | 179.6 | 916 | 9,206 |

- **Half update speed regained** at k½ = ln((κ−1)/3)/(−ln β2). For κ = 10⁴ that is 158.2, 806 and 8,104
  steps [I]. This matches RESEARCH §4.1's 158 at 0.95 [V].

### 1.3 Data spike (gradient g² = κ·ḡ² for w consecutive steps)
Excess after the window, k steps later: v/ḡ² − 1 ≈ (κ−1)·β2^k·(1−β2^w). The factor (1−β2^w) is 0.05 for
w = 1 and 0.226 for w = 5 at 0.95. "Within 2×" therefore comes at k ≥ ln((κ−1)(1−β2^w))/(−ln β2):

| (w, κ) | k at β2 = 0.95 |
|---|---|
| (1, 10²) | 31 |
| (1, 10⁴) | 121 |
| (5, 10²) | 61 |
| (5, 10⁴) | 151 |

At β2 = 0.999 the excess is roughly **6.5× longer-lived** than these numbers suggest from the decay rate
alone, because each spike step adds less (factor 1−β2 = 0.001) but stays far longer. The net effect still
favours β2 = 0.999 for persistence [I].

### 1.4 What this means for the design
1. **The thesis depends on β2.** At β2 = 0.95 the optimizer state forgets in ~100–200 steps, so H1's
   "state-carried share" must be measured at cut times inside that window. That is why H1 is defined at
   t1 = s1+10. At β2 = 0.999, state damage can persist through a whole 3,000-step stress run.
2. **New design lever (recommended): treat β2 as an instrument.** Run the damage-migration curve at
   β2 ∈ {0.95, 0.99, 0.999}. H2 then becomes a **sharp quantitative prediction**: the state-carried share
   should decay at rate ≈ −ln β2 per step, for example a half-life of 13.5 / 69 / 693 steps. A matched slope
   across three β2 values is strong mechanistic evidence and hard to fake. A mismatch tells us the damage
   isn't simply v-memory. Cost: 3× the migration-curve forks, about 1–3 GPU-h at proxy [A].
3. **m is irrelevant beyond ~30–50 steps** (e-fold 9.5). An m-only effect at t1 ≥ s1+50 should be ≈ 0. That
   gives a free sanity prediction.

---

## 2 · Update-size consequences (why the baselines are or aren't fair)

### 2.1 v-only reset with the step counter kept (current `Bvreset` / `Bg`) [V: reset.py keeps `step`]
- At large t, bias correction is ≈ 1. After v ← 0, the next step gives v = (1−β2)g², so the update on each
  coordinate is ≈ m/(√(1−β2)·|g|), which is **1/√(1−β2) = 4.47×** the normal size (at 0.95).
- This is the documented "~4.5× overshoot" [V: reset.py docstring; RESEARCH §4.1].
- A **full reset** (m ← 0 too) gives update ≈ (1−β1)g/(√(1−β2)|g|) = 0.447·sign(g) on the first step, then
  grows. It still deviates from normal Adam until bias correction re-equilibrates, which it never does,
  because the step counter is large.

### 2.2 Adam-MR (m, v and t all reset; Asadi 2023)
- With t ← 0 the bias correction is 1/(1−β^t), so the first update is m̂/√v̂ = g/|g| = ±1 per coordinate:
  bounded, but at full step size on every coordinate (the RAdam variance problem).
- **This is the fair "full reset".** Our `Bg` without a t reset is mis-specified by construction (Skeptic
  M5). Proposed: add `Bg_t` (= Adam-MR) and report both.

### 2.3 Why clipping only softens spikes: the anti-windup invariance (new; derived)
- Take a coordinate where one spike gradient G dominates (G² ≫ ḡ²/(1−β2)). After the spike, with small
  gradients:
  - m_k ≈ (1−β1)·β1^k·G
  - v_k ≈ (1−β2)·β2^k·G²
  - update_k/η = m_k/√v_k = [(1−β1)/√(1−β2)]·(β1/√β2)^k = **0.447·0.9234^k**, **independent of G**.
- **Total post-spike drift** on that coordinate: Σ_k 0.447·0.9234^k·η = **5.84·η**, all in the sign of
  G [I].
- **Consequences:**
  - Once a coordinate is spike-dominated, *how big* the spike was barely matters. Clipping G changes the
    magnitude but not the ≈ 5.8-learning-rate drift. That explains Marin's "clipping softened spikes but
    never removed them" mechanistically.
  - Clipping helps only by keeping coordinates *below* the dominance threshold.
  - The dominance condition (1−β2)·β2^k·G² ≫ ḡ² sets how long this lasts: k < ln((1−β2)G²/ḡ²)/(−ln β2).
- **Predictions (add as H10):**
  1. **(a) Plateau:** damage in weight space saturates in corruption magnitude κ above the dominance
     threshold, so dose-response (H3) should plateau, not grow without bound.
  2. **(b) Clip invariance:** the post-spike update RMS on dominated coordinates is within ±10% between
     clip 1.0 and 0.2 while dominance holds (T9 in `02_adjacent_fields.md`).
  3. **(c) Undo at lag L removes the remaining drift 5.84·η·0.9234^L,** so the undo benefit should decay
     with lag at rate ln(1/0.9234) = 0.0797 per step (half-life 8.7 steps). That is a sharp,
     β-determined prediction for H6.
  4. **(d) Undo of v alone is predicted to hurt.** It shrinks v while m stays large, so updates *grow*.
     Undo of m alone should help, and undo of both should come closest to the oracle. This is a falsifiable
     ordering for the H6 ablation.

### 2.4 Hybrid feasibility bound (for patch hygiene)
- By Cauchy–Schwarz on the two EMAs: |m_t| ≤ (1−β1)·Σ_k β1^k|g_{t−k}| ≤ (1−β1)·√(Σ_k β1^{2k}/((1−β2)β2^k))·√v_t.
- So |m| ≤ C·√v with **C = (1−β1)/√(1−β2) · 1/√(1−β1²/β2)**, valid when β1² < β2. At (0.9, 0.95) that gives
  0.4472 × 2.604 = **1.165** [I; it matches `02_adjacent_fields.md` T8].
- Honest Adam states always satisfy it. A hybrid (m from B\*, v from B0) can violate it where v^0 ≪ v^\*.
- **Count violations on every hybrid.** A high violation rate means the hybrid is off-manifold and its Δ is
  not interpretable as a natural counterfactual.

---

## 3 · The undo log: exactness and numerics

- **Exactness.** m and v are linear in {g_s, g_s²} with fixed coefficients. Nothing feeds back from w into
  the recurrences, apart from *future* gradients, which is the indirect path. So the direct contribution of
  window W to (m, v) at t1 is exactly:
  - c_m = (1−β1)·Σ_{s∈W} β1^{t1−s}·g_s
  - c_v = (1−β2)·Σ_{s∈W} β2^{t1−s}·g_s²

  Subtracting them is an exact **state-only retroactive skip**. Weights and later gradients are untouched.
- **Replacement term.** Replace the removed g² by a clean expectation: v ← v − c_v + (1−β2)·Σ_{s∈W}
  β2^{t1−s}·v̄. Otherwise v drops below its natural level and we risk under-preconditioning (sign −).
- **Non-negativity.** In exact arithmetic v_{t1} − c_v ≥ β2^{t1−s0+1}·v_{s0−1} ≥ 0. In fp32, rounding can
  give −1 ulp, so clamp at 0. The replacement term usually dominates anyway.
- **Use recorded gradients, not snapshot differences, in the experiments.** Reconstructing
  (1−β2)·g_s² = v_s − β2·v_{s−1} from fp32 snapshots has absolute error ~ulp(v_s) ≈ 6e-8·v_s:
  - On spike coordinates (g² ≫ v) the relative error is negligible.
  - On quiet coordinates the relative error is large, but those terms are tiny.

  The **experiment** should record post-clip gradients for s ∈ W directly in a `pre_opt_step` hook (exact;
  no harness change). The **deployable** airbag can use the reconstruction.
- **Memory.** A rolling gradient buffer of length |W| costs |W|×params×4 bytes: 44 MB per step at 11M and
  0.5 GB per step at 124M. A 5-step window at 124M is 2.5 GB, trivial on 192 GB [I].
- **What undo can't fix:**
  - weight damage already taken (the indirect path);
  - spikes not caused by abnormal gradients (`lr_bump`: the gradients are normal and the step size isn't,
    so c_m and c_v are ordinary terms whose removal just deletes useful history; H6 predicts < 20%
    recovery);
  - planted state edits (`v_inflate`: no gradient caused them, so undo is a no-op; the parity residual T8
    flags them instead).

---

## 4 · Mediation estimands: exact definitions (fixes the H1 wording)

**Setup.** Cell (a_w, a_m, a_v), with 1 = taken from B0 (damaged) and 0 = taken from B\*. D(abc) = Δ of that
hybrid vs B\* at the chosen horizon. D(000) = 0 and D(111) = Δ_B0.

**Main and interaction effects (Möbius):**

| Effect | Formula |
|---|---|
| e_w | D(100) |
| e_m | D(010) |
| e_v | D(001) |
| e_wm | D(110) − D(100) − D(010) |
| e_wv, e_mv | (analogous) |
| e_wmv | D(111) − D(110) − D(101) − D(011) + D(100) + D(010) + D(001) |

**Two natural state shares, which differ by the w×state interactions:**
- **Noising (sufficiency)**: N_state = D(011)/D(111). Is the damaged state alone enough to cause the damage?
- **Denoising (necessity)**: R_state = 1 − D(100)/D(111). Does repairing the state recover?
- N_state − R_state = −(e_wm + e_wv + e_wmv)/D(111).

**Recommended single number for H1:** the 2-player Shapley share φ_state/D(111), where
φ_state = ½·[D(011) + D(111) − D(100)]. It is the average of the two shares, symmetric and standard, and it
is the H1 endpoint we propose to pre-register. Always report N, R and the interaction alongside it.

**The v/m split inside the state:** report e_m, e_v and e_mv with the cut-time sweep.

**What Xu 2026 found (abstract):** m carried ~90% and v ~0% in subliminal transfer. Our prediction differs
by spike type (§7), which is a direct, citable contrast.

---

## 5 · Statistics and power

**Unit, clustering and design effect.** The unit is the instance. The cluster is (init seed × data-order
seed). With 2 sites per seed and intraclass correlation ICC, the design effect is 1 + (2−1)·ICC, so n_eff =
n/(1+ICC). At ICC = 0.5, 10 instances behave like 6.7. **Use data-order seeds** so each instance has its own
cluster (n_eff ≈ n).

**Paired superiority, 80% power, two-sided α = 0.05:** n ≈ (1.96 + 0.84)²/d_z² + 1 ≈ 7.85/d_z² + 1, where
d_z = mean paired Δ / SD of paired Δ [I]:

| d_z | n |
|---|---|
| 1.5 | 5 |
| 1.0 | 9 |
| 0.8 | 13 |
| 0.5 | 33 |

**TOST non-inferiority / equivalence (true difference 0, margin δ, SD σ of paired differences):**
n ≈ (z_{0.95} + z_{0.90})²·(σ/δ)² ≈ 8.6·(σ/δ)² [I]:

| σ / δ (δ = 0.005 nats) | n |
|---|---|
| 0.4 (σ = 0.002) | 2 |
| 0.8 (σ = 0.004) | 6 |
| 1.4 (σ = 0.007) | 17 |
| 2.0 (σ = 0.01) | 35 |

So **the pilot SD decides everything.** If σ ≥ 0.007 nats, widen the margin (≥ 2 × the placebo spread, per
G3) or use 15–20 instances for the primary contrast. Proposal: pre-register the rule "n = max(10,
⌈8.6(σ̂/δ)²⌉) using the pilot σ̂". That is pre-specified, so it is legitimate.

**Multiplicity.** One primary test (H7). Holm across H1–H6 and H8–H10 (10 tests). BH-FDR (q = 0.1) across
exploratory ablations. Never pool spike types.

**Oracle stability.** Group testing with threshold τ_f: report the Jaccard overlap of S\* across 2 placebo
seeds. Below 0.7, switch to SuSiE credible sets (`02_adjacent_fields.md` T3).

---

## 6 · Cost model per scale

[V] measured on Kaggle: proxy 0.0619 s/step at batch 16 on a P100; 124M 64–68 s per optimizer step on a
P100 at grad_accum 40 (`step_timer_results.md`, V7 §3). [A] MI300X speedups are assumed. `step_timer` on
AMD day 1 replaces every number below.

| Scale | Tokens/step | Realistic trunk (≥ 20 tok/non-emb param, warmup+cosine) | MI300X step (2× / 10× / 18× of P100) | Trunk time | 300-step fork |
|---|---|---|---|---|---|
| Proxy 3L/192 (1.33M non-emb) | 64×256 = 16,384 | 26.6M tokens ≈ 1,630 steps | (0.25 s @ batch 64 on P100 [V-ish]) → 0.12 / 0.025 / 0.014 s | 3.3 / 0.7 / 0.4 min | 37 / 7.5 / 4 s |
| Mid 6L/384 (≈10.6M non-emb) [I] | 16,384 | 212M tokens ≈ 12,900 steps | est. 0.3 / 0.06 / 0.035 s [A] | 65 / 13 / 7.5 min | 90 / 18 / 10 s |
| 124M (85M non-emb) | 8×1024×40 = 327,680 | 1.0B tokens ≈ 3,050 steps (a 12 tok/param compromise) | 33 / 6.6 / 3.7 s | 28 / 5.6 / 3.1 h | 2.8 / 0.55 / 0.31 h |

**Experiment-level estimates at the 10× scenario [A]:**
- **Proxy:** full battery on realistic trunks with ~10 instances per type × 4 types. Per instance: 8
  factorial cells × 3 cut times, ~12 repair/baseline branches, 5+5 placebo, and a ~15-fork oracle, so about
  60 forks. That is ~2,400 forks × 7.5 s ≈ **5 GPU-h**. With the β2-instrument sweep (§1.4), +3 GPU-h.
- **Mid rung:** key branches, ~20 forks × 20 instances × 18 s ≈ **2 GPU-h**, plus trunks (~10 × 13 min ≈
  2.2 h).
- **124M primary contrast**, n = 3 (n = 5 if possible). Per instance: 4 factorial cells (B0, B\*,
  state-only, weights-only), ~4 repair/baseline branches, and 2 placebo, so 10 forks at horizon 300.
  That is 10 × 0.55 h ≈ 5.5 h per instance → **16.5 h for n = 3** (27.5 h for n = 5), plus 3 trunks ×
  5.6 h = 17 h. Total about **34–45 GPU-h**.
- **Grand total:** about 50–60 GPU-h at 10×. That is 1.5–3× more at 2–5×, which is why `step_timer` is
  first, and why at 124M the factorial is cut to 4 cells and the horizon to 300 steps (pre-registered
  as a secondary horizon there).

**Concurrency.** A 192 GB MI300X can hold many proxy forks at once. Running k forks in separate processes
has to pass the cross-process determinism gate first (M12). If it passes, the proxy wall-clock drops by
up to k×.

---

## 7 · A-priori predicted damage maps per spike type (written down before any data)

| Type | Mechanism (expected) | Poison sign σ | Where | Predicted state share φ at s1+10 (β2 = .95) | Undo (H6) | Best repair |
|---|---|---|---|---|---|---|
| `corrupt_batch` (uniform random tokens, width 5) | abnormal gradients on many wte rows (rare tokens get large sparse gradients) plus all blocks; m/v dominated on those coordinates (§2.3) | **+** (over-preconditioned) on hit rows | wte rows of sampled tokens; ln / c_attn moderately | **high** (≥ 0.3) | ≥ 70% of oracle at lag ≤ 5; decays with half-life ~8.7 steps | undo (m+v) ≈ restore-to-v̄ on hit rows |
| `repeat_ngram` (OLMo cause class) | a few token rows and positional patterns get coherent large gradients | + on few rows | wte rows of the repeated tokens; attention in early layers | high and **sparse** (\|S\*\| ≤ 5% of rows) | high | row-level undo / restore |
| `lr_bump` (10× LR, width 5) | gradients normal at onset; damage enters **weights** through big steps; loss rises afterwards, so v inflates later | ~0 at onset, + later | diffuse | **low** (≤ 0.15) | **< 20%** | weight damage mostly; selective state repair helps little, rollback is competitive |
| `v_inflate` (planted, κ = 10⁴) | pure state | + on targets | named targets only | ≈ 1.0 at s1+1 (harness sanity G1) | **no-op** (no gradient cause) | restore v_S ← v̄_S (full fix, B5) |
| natural, β2 = 0.999 stress | Bai decay stage (v ≪ g²) **before** onset, refill **after** | **− before, + after** | blocks where v decayed (Bai: embeddings / W_V) | moderate; long memory | partial | **certified restoration** (needs the sign) |
| natural, β2 = 0.95 stress | attention-logit growth possible (Wortsman, Marin) | mixed | attention (c_attn) | uncertain; QK-Norm ablation decides | partial | report |

These predictions are deliberately specific. Several could fail, and that is the point.

---

## 8 · Research-plan risk register

| Risk | P | Impact | Mitigation / early signal |
|---|---|---|---|
| ROCm can't reach bitwise Δ = 0 | medium | high | A6 checklist; cross-process gate week 1; tolerance path; BACKUP 2 needs no forks |
| MI300X much slower than assumed on tiny shapes | medium | medium | `step_timer` first; concurrency; trim the factorial at 124M |
| H1 refuted (state share ≈ 0) | low–medium | medium | pivot insurance (negative-result framing) |
| AtP-state fails vs the oracle (Adam nonlinearity) | medium | medium | λ-decomposition variant (BACKUP 1); report the failure |
| A rival preprint on spike-state localization before Jan | low–medium | high | file the related-work table early; arXiv preprint ~2 weeks before the deadline |
| Bus factor 1 (spike engineer) | high | high | assign C2/C3/analysis owners this week (improvement C6) |
| Natural-spike B\* undefined | certain | medium | pre-register option (b): exclude natural spikes from causal contrasts |
| ICML 2027 date unannounced | certain | low | plan to Jan 22; re-check monthly |

---

## 9 · Proposed PLAN_V7 §4.7 entries (consolidated, for sign-off)

| Item | Value |
|---|---|
| τ (kill-test) | 0.01 nats |
| Primary endpoint | H7 gap closed at end of schedule (proxy, mid); h = 300 at 124M |
| H1 endpoint | Shapley state share φ_state/D(111) at t1 = s1+10 |
| Non-inferiority margin | max(0.005 nats, 2 × placebo spread) |
| n rule | max(10, ⌈8.6(σ̂/δ)²⌉) from the 4-instance pilot |
| Seeds | 0 tune; 1–9 held out; data-order seed per instance |
| Δ eval set | val batches ≥ 1000, ≥ 1M tokens, fixed |
| `Bs` clip | 0.5 × trunk grad_clip |
| Fair full reset | report both Bg (t kept) and Adam-MR (t reset) |
| Natural spikes | excluded from causal contrasts; used for localization and detection |
| β2 instrument | migration curve at β2 ∈ {0.95, 0.99, 0.999} (H2 slope ≈ −ln β2) |
| Oracle | group testing with τ_f = max(0.005, 5 × placebo spread); SuSiE fallback if Jaccard < 0.7 |

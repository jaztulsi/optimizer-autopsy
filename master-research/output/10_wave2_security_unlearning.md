# 10 · Wave 2: security, distributed, unlearning and fault-tolerance angles (2026-10-22)

*Wave-2 researcher role. Web research and reading only. No code was executed; the only shell command used for
research was `date`. Builds on `02_lineage.md`, `02_adjacent_fields.md` (T1 undo log, T8 parity residuals, T16
Byzantine aggregation), `03_novelty_matrix.md` v2 and `05_shortlist.md`. Items those files already cover are
only cross-referenced here.*

**Labels.** **[verified]** = I read it this session (abstract, full text or official record; locator given).
**[verified-snippet]** = confirmed only from a search-engine snippet or catalog page. **[inferred]** = my
reasoning from what I read. **[assumed]** = not checked. **UNVERIFIED** = could not confirm, so it stays out of
every claim.

**Decision this serves.** Two questions. (a) Does a security, systems or unlearning literature already own
"optimizer-state poisoning", "exact undo of Adam updates" or "repair instead of rollback"? That would downgrade
K3, K4 and the undo log [5]. (b) Does such work open a cheap new framing or application for the PRIMARY direction
(w/m/v damage migration plus undo / certified repair)?

## Bottom line

1. **The biggest finding is a correction, not a new idea.** **SWIFT** (Zhong et al., arXiv
   [2302.06173](https://arxiv.org/abs/2302.06173), 2023-02-13; IEEE TPDS 35(9), Sep 2024) already inverts
   Adam's and AdamW's update in closed form:
   - it computes m_{t−1} = (m_t − (1−β1)g′_t)/β1 and v_{t−1} = (v_t − (1−β2)g′_t²)/β2 from the cached last
     gradient;
   - it uses this to "undo" partially applied steps during failure recovery.

   [verified: ar5iv full text, §"update-undo", Algorithms 6 and 8, Table I]. So `02_adjacent_fields.md` T1 (undo
   log, verdict **NEW**, novelty 5) must drop to **PARTIAL**. The mechanism (Adam's moment recurrences are
   invertible given g) is prior art. Our version still differs in four ways:
   - it is retroactive: it removes a *past window's* contribution after later steps have run;
   - it edits state only, leaving the weights alone;
   - it is computed forward, with no division by β;
   - it is used for causal path decomposition and repair.
2. **No paper treats Adam's `m`/`v` as an attack target.** Across 30 queries (§7), no paper crafts gradients
   or data to inflate or deflate Adam's moments, or attacks server-side FedAdam/FedYogi moments by design. The
   nearest work:
   - Altenbernd 2026: SDC drives v→∞ and freezes parameters, but by accident, not as an attack;
   - one book chapter that tests model poisoning against FedAdam/FedYogi/FedAdagrad (content not assessed);
   - time-coupled Byzantine attacks, which momentum is used to *defend* against.

   So the framing "**loss spikes are natural availability attacks on optimizer state; v-inflation freezes,
   v-deflation destabilizes**" looks new as a framing [inferred, bounded by §7]. It costs no compute.
3. **Unlearning has state-exactness, but not for Adam in a cheap form.** Three precedents:
   - Unlearning-at-Scale (2508.12220) is bit-exact in model and optimizer state, but only by full replay;
   - Stewart 2026 (two papers) shows that L-BFGS optimizer state keeps "residual information, supposedly
     deleted";
   - Amnesiac ML subtracts weight updates only.

   No one subtracts a window's contribution from Adam's m/v [verified abstracts; kill-search §7]. The undo log is
   best pitched as **"state-only retroactive deletion"**, with SWIFT cited as the mechanism's ancestor.
4. **Repair-instead-of-rollback exists for compute faults, not for optimizer state.**
   - ATTNChecker (ABFT for attention, PPoPP 2025) corrects INF/NaN in place and reports up to 49× less recovery
     overhead than checkpoint/restore [verified-snippet].
   - CPR (MLSys 2021) does partial recovery that mixes stale and fresh shards [verified-snippet].
   - FedRecover, Crab and FAUN repair poisoned FL *weights* without retraining [verified abstracts].

   None repairs m/v. Altenbernd 2026 lists momentum resets as future work (`02_lineage.md` row 5).
5. **Deterministic replay as a diagnostic is industry practice.**
   - Gemini (arXiv 2312.11805, §3): "techniques that exploit deterministic replay to isolate incorrect
     computations" [verified].
   - Ma et al. 2025: XLA-deterministic healthy-vs-unhealthy node comparisons [verified].
   - SCOUT 2026: replay "certifies checkpoint numerical integrity" [verified abstract].

   This reinforces the existing rule: claim the *counterfactual instrument on optimizer state*, not determinism
   or replay.

---

## 1 · Ranked findings (prior art and threats)

**Columns.** Overlap is scored against the claims as `03_novelty_matrix.md` defines them:
- **K1** instrument; **K2** localization; **K3** repair; **K4** state patching; **U** undo log [5].

**Threat** is to *our* novelty claims.

| # | Work (link · date) | Read depth | What it does | Overlap | Threat | Novel framing it enables |
|---|---|---|---|---|---|---|
| 1 | Zhong, Sheng, Liu, Yuan, Wu, **SWIFT: Expedited Failure Recovery for Large-scale DNN Training**. [2302.06173](https://arxiv.org/abs/2302.06173) · 2023-02-13; IEEE TPDS 35(9), 2024 | full text (ar5iv) [verified] | Recovers from failures without extra state copies. **"Update-undo"** inverts SGD-momentum, Adam, AdamW and LAMB (LAMB by saving the scalar norm) using only the cached latest gradient. Notes that "the recovered state may slightly differ from the original state due to floating-point errors". AMSGrad is "not applicable". Falls back to logging plus replay. | **U: high** (the mechanism). K3: low | **medium** (high if we keep saying "exact undo, NEW") | Cite it as the ancestor. Our delta: a retroactive window, state only, forward compensation with no 1/β amplification [inferred], used for path decomposition. Their AMSGrad caveat matters if `v ← max(v, v̄)` repairs are used. |
| 2 | Xu, **Stored in Optimizer State, Valued by Later Training**. [2608.20442](https://arxiv.org/abs/2608.20442) · 2026-08-20 | abstract [verified] | m is the causal carrier of a subliminal trait. "Transplanting it alone leaves parameters, hidden states, and outputs unchanged at the cut". **No security framing**. | K4: high (already in v2) | medium (known) | It is the non-adversarial proof that optimizer state can carry a dormant payload. That opens an **optimizer-state supply-chain** threat model (S3). |
| 3 | Ma, Pei, Lausen, Karypis, **Understanding Silent Data Corruption in LLM Training**. [2502.12340](https://arxiv.org/abs/2502.12340) · 2025-02-17 (ACL 2025 per [anthology preview](https://preview.aclanthology.org/landing_page/2025.acl-long.996.pdf), verified-snippet) | HTML, first 100k chars [verified] | Real unhealthy nodes vs healthy ones. XLA "fully deterministic instruction ordering". Synchronizes submodule outputs and parameters to isolate SDC at three levels: submodule, one optimizer step, training period. Finds that weights "incrementally drift", and that fine-tuning loss spikes can lead to 0 accuracy. **Does not discuss optimizer state** (it broadcasts parameters only). | K1: medium (paired deterministic runs as a fault-isolation instrument) | low–medium | Their parameter-only sync leaves the unhealthy node's m/v unsynchronized [inferred]. Our w/m/v split is exactly the missing control: does SDC damage persist through m/v after the weights are re-synced? |
| 4 | Altenbernd, Wiesner, Kao, **Exploring SDC as a Reliability Challenge in LLM Training**. [2604.00726](https://arxiv.org/abs/2604.00726) · 2026-04-01 | (see `02_lineage.md` row 5) | Corrupted gradient → v→∞ → frozen params. Recompute-step detector. Momentum resets listed as future work. | K3/U: medium | medium (known) | S4: localized repair-instead-of-rollback for SDC. |
| 5 | Stewart, **Shape of Memory: … Machine Unlearning in Second-Order Optimizers**. [2604.23046](https://arxiv.org/abs/2604.23046) · 2026-04-24; Stewart, **Form and Function …** [2605.17590](https://arxiv.org/abs/2605.17590) · 2026-05-17 | abstracts [verified] | Second-order (L-BFGS) learners match the counterfactual on performance and gradients, yet "the second-order optimizer shows significant volatility in the optimizer state … residual information, supposedly deleted". Proposes parameter, memory and update-direction error metrics against a counterfactual state. | U, K4: medium (concept) | low–medium | S8: an "Adam residual-memory audit". Their metric triple maps onto our ‖Δw‖, ‖Δm‖, ‖Δv‖ vs B\*. Convex/L-BFGS only. |
| 6 | X, **Unlearning at Scale: State-Exact Trace-Preserving Deletion**. [2508.12220](https://arxiv.org/abs/2508.12220) · 2025-08-17 (v2 2026-08-12) | abstract [verified] | Replay without the deleted rows "reconstructs a separately executed trace oracle bit-for-bit in model and optimizer state". Pythia 160M/2.8B and Llama-3.2-1B, pinned single-GPU setup. Scattered deletions force near-full replay. | K1: low (already conceded). U: low | low | It is the expensive *exact* end of the spectrum. The undo log is the cheap, state-only, *direct-path* end. Our forks measure the gap between them. |
| 7 | Graves, Nagisetty, Ganesh, **Amnesiac Machine Learning**. [2010.10981](https://arxiv.org/abs/2010.10981) · 2020-10-21 (AAAI 2021) | secondary summaries [verified-snippet] | Stores per-batch parameter updates and subtracts them. Efficacy falls as more batches are affected. A 2025 paper says it "has never been implemented for LLMs" ([2506.20941](https://arxiv.org/abs/2506.20941)). | U: medium (concept, weights only) | low | Already in T1. It is the weight-side twin of the undo log. |
| 8 | Maclaurin, Duvenaud, Adams, **Gradient-based Hyperparameter Optimization through Reversible Learning**. [1502.03492](https://arxiv.org/abs/1502.03492) · ICML 2015 | ar5iv §2.2–2.3 [verified] | Exactly reverses SGD-momentum. Multiplying by γ<1 "destroy[s] the least significant bits". Naive reversal errors "accumulate exponentially", so they store the lost bits (≈0.029 bits/step at γ=0.98). | U: precision caveat | low | Warns that *backward* undo (SWIFT-style ÷β per step) amplifies rounding. Our forward compensation avoids it [inferred]. This needs the CPU unit test T1 already requires. |
| 9 | Cao, Jia, Zhang, Gong, **FedRecover**. [2210.10936](https://arxiv.org/abs/2210.10936) · 2022-10-20 (IEEE S&P 2023); Jiang et al., **Crab** [2401.08216](https://arxiv.org/abs/2401.08216) · 2024-01-16; Zhao et al., **FAUN** [2605.02110](https://arxiv.org/abs/2605.02110) · 2026-05-04 | abstracts [verified] | Repair a poisoned FL global model **without retraining**, from stored history (FedRecover), selective history plus "adaptive model rollback" with a bound vs retrain (Crab, "certified"), or a short window of malicious updates cancelled adversarially (FAUN). **No abstract mentions optimizer state, momentum or Adam.** | K3: medium (concept: post-attack repair, not rollback) | low | S5: these recover *weights only*. With FedAdam/FedYogi the server's m/v keep the attackers' contribution [inferred]. **Terminology clash:** Crab already says "certified recovery". Our "stability-certified" must be named distinctly. |
| 10 | Nabavirazavi, Taheri, Ghahremani, Iyengar, **Model Poisoning Attack Against Federated Learning with Adaptive Aggregation**. In *Adversarial Multimedia Forensics*, Springer AIS vol. 104, pp. 1–27, doi [10.1007/978-3-031-49803-9_1](https://doi.org/10.1007/978-3-031-49803-9_1) · 2023-11-15 | catalog abstract [verified-snippet] | Tests model poisoning against FedAdagrad, FedYogi and FedAdam on image data. Whether it analyzes server moments: **not assessed**. | K3: low | low | The only work found that attacks adaptive server optimizers. Its mechanism (moments or not) is UNVERIFIED. |
| 11 | El-Mhamdi, Guerraoui, Rouault, **Distributed Momentum for Byzantine-resilient SGD**. [2003.00010](https://arxiv.org/abs/2003.00010) · 2020-02-28 (ICLR 2021); Karimireddy, He, Jaggi, **Learning from History for Byzantine Robust Optimization**. [2012.10333](https://arxiv.org/abs/2012.10333) · 2020-12-18 (ICML 2021) | abstracts [verified-snippet] | Worker momentum *helps* Byzantine robustness by lowering the variance-to-norm ratio. Attackers who "couple their attacks across time" defeat per-round robust aggregators. Fix: iterative clipping plus worker momentum. | K1 (framing) | low | S6: a **tension** to state. Byzantine work treats momentum as a defense; we treat m/v as a carrier of accepted damage. Both can be true [inferred]. |
| 12 | Shumailov et al., **Manipulating SGD with Data Ordering Attacks**. [2104.09667](https://arxiv.org/abs/2104.09667) · 2021-04-19 (NeurIPS 2021) | ar5iv [verified] | Batch reorder, reshuffle and replace attacks (BRRR). Integrity and availability effects: "An attack at just one epoch is enough to degrade the training for more than 90 epochs". Uses Adam and SGD (Tables 1, 4, 5). **Attributes the effect to data order, not to optimizer state.** | K6 (spike recipes) | low | S1/S2: an ordering attack is a data-preserving way to *induce* spikes. Our forks can say whether its long-lived damage lives in w or in m/v. Nobody has measured that. |
| 13 | Bouaziz, El-Mhamdi, Usunier, **Inverting Gradient Attacks Makes Powerful Data Poisoning**. [2410.21453](https://arxiv.org/abs/2410.21453) · 2024-10-28 (TMLR 2025) | abstract [verified] | Gradient inversion turns malicious *gradients* into data points. Availability attack to random-level accuracy with about 1% poisons. | — | low | S2: any gradient-space state attack, such as targeted v-inflation, is in principle data-realizable [inferred]. |
| 14 | Liang et al., **ATTNChecker: Highly-Optimized Fault Tolerant Attention for LLM Training** [2410.11720](https://arxiv.org/abs/2410.11720) · 2024 (PPoPP 2025, doi 10.1145/3710848.3710870) | search snippets [verified-snippet] | ABFT for attention. INF/NaN/near-INF faults "trigger non-trainable states". Detects and corrects them in place at about 7% overhead, with up to 49× lower recovery overhead than checkpoint/restore. | K3 (repair-not-rollback, for compute) | low | It is the compute-side precedent for "repair instead of rollback". Nobody does the optimizer-state side. |
| 15 | He, Hutton, Chan, De Gruijl, Govindaraju, Patil, Li, **Understanding and Mitigating Hardware Failures in Deep Learning Training Systems**. ISCA 2023, doi [10.1145/3579371.3589105](https://doi.org/10.1145/3579371.3589105) | record only [verified-snippet] | Described as the first in-depth resilience study of DNN training under accelerator logic failures. Ma 2025 cites the same group for Inf/NaN and loss-spike signatures of SDC. | — | low | Background for S4. Content not assessed. |
| 16 | Gemini Team, **Gemini** [2312.11805](https://arxiv.org/abs/2312.11805) §3 · 2023-12-19; Llama Team, **Llama 3 Herd** [2407.21783](https://arxiv.org/abs/2407.21783) · 2024-07-31 | Gemini HTML [verified]; Llama via Ma 2025's citation [verified-secondary] | Gemini: "we can expect SDC events to impact training every week or two"; deterministic replay; "proactive SDC scanners on idle machines and hot standbys". Llama 3: "6 unplanned job interruptions were attributed to SDC during a 54-day pre-training snapshot" (quoted by Ma 2025). | K1 wording | low | SDC is a production-rate event. It is a second, non-data natural source of state poisoning (S4). |
| 17 | LLM-PRISM [2604.10390](https://arxiv.org/abs/2604.10390) · 2026-04-12; **TrainSDC** [2608.30769](https://arxiv.org/abs/2608.30769) · 2026-08-31; **SCOUT** [2608.11034](https://arxiv.org/abs/2608.11034) · 2026-08-11 | abstracts [verified] | LLM-PRISM: 7,664 Megatron runs with fault injection across FP16/BF16/FP8. TrainSDC: backward-pass SDC impact "depends mainly on gradient exponent distributions"; three protections at 1.65–6.76% overhead. SCOUT: replica-consensus failure localization; "clean replay coverage certifies checkpoint numerical integrity". **None mentions optimizer state** in its abstract. | K1 (replay certification: SCOUT) | low | TrainSDC's exponent finding is consistent with gradient-exponent faults hitting v [inferred], which motivates the `sdc_grad` recipe in S4. |
| 18 | Maeng et al., **CPR: … Partial Recovery**. [2011.02999](https://arxiv.org/abs/2011.02999) · MLSys 2021 | search snippets [verified-snippet] | After a failure, healthy nodes continue while failed embedding shards reload stale checkpoints: a deliberately *hybrid* state with a measured accuracy cost. Emulated by clearing 12.5/25/50% of tables. | K4: low (concept) | low | A systems precedent for "mixed-time hybrid states are usable". K4's cross-world hybrids are the causal version. |
| 19 | Li et al., **ExCP: Extreme LLM Checkpoint Compression via Weight-Momentum Joint Shrinking**. [2406.11257](https://arxiv.org/abs/2406.11257) · ICML 2024 (PMLR 235) | search snippets [verified-snippet] | Prunes checkpoint residuals jointly with optimizer momentum. About 70× on Pythia-410M. | K2: low | none–low | Lossy optimizer state is benign "poison". C2's oracle maps could say which m/v coordinates are safe to drop (S10, off-scope). |
| 20 | **NeuroImprint** [2606.20553](https://arxiv.org/abs/2606.20553) · 2026-06-18; **TriShield** [2607.27940](https://arxiv.org/abs/2607.27940) · 2026-07-30 | abstracts [verified] | Privacy backdoor in federated PEFT that must handle "stateful optimizers (e.g., Adam/AdamW)" cross-step mixing. TriShield's defense "forces Adam/AdamW's momentum state to irreversibly entangle gradients". | — | none | Optimizer state appears in security papers only as a *mixing nuisance or tool*, never as the target. |
| 21 | Rahaman et al., **(A)iSpy: Parasitic Trojans for ML Infrastructure**. [2607.17550](https://arxiv.org/abs/2607.17550) · 2026-07-20 | abstract [verified] | A trojan in the compute graph (ONNX Runtime) that "acts as a gradient amplifier". It turns weak poisoning into a backdoor, and its appendix covers availability disruption. **The optimizer state is not mentioned in the abstract.** | — | low | This threat model (a compromised runtime that can touch tensors in the step) is the one in which a *direct* m/v write is realistic (S3). |
| 22 | Cinà et al., **Sponge Poisoning** [2203.08147](https://arxiv.org/abs/2203.08147) · 2022-03-14 | snippets [verified-snippet] | A training-time attack on energy and latency, not accuracy. | — | none | Precedent that attackers target *non-accuracy* properties of training. |
| 23 | Habani, Refael, Lindenbaum, **Which Optimizer Leaks Most?** [2610.18294](https://arxiv.org/abs/2610.18294) · 2026-10-13 | abstract [verified] | Optimizer implicit bias controls training-data reconstruction from released *weights*. Adam is most resistant to the attacks tested. Moments are not studied. | — | none | S8: whether moment buffers leak is open (kill-search §7 Q22). |
| 24 | Jia et al., **Proof-of-Learning** [2103.05633](https://arxiv.org/abs/2103.05633) · 2021-03-09 | abstract [verified] | Training-trajectory proofs by replaying logged steps. | K1 (replay audit) | none | Our edits plus Δ==0 forks could make *auditable* state repairs. Not pursued. |

*Row 14 note:* the first author "Liang" (Yuhang Liang) comes from the search-result PDF filename and author
listings, not from the paper itself [verified-snippet].

---

## 2 · Per-topic findings

### 2.1 Attacks that target optimizer state (question 1)
- **No dedicated literature found [verified, bounded by Q1–Q9, Q24–Q27 in §7].** No paper crafts gradients or
  data to inflate or deflate Adam's moments. No paper attacks FedAdam/FedYogi server moments *as moments*. No
  paper shows a backdoor carried in a shipped `optimizer.pt`.
- **Closest neighbours.**
  - **Accidental state poisoning:** Altenbernd 2026 (v→∞ freezes parameters).
  - **Adaptive aggregators attacked:** Nabavirazavi et al. 2023 (content not assessed).
  - **Adam as camouflage:** with client-side Adam, Byzantine clients "can camouflage their model parameters"
    ([2006.08259](https://arxiv.org/abs/2006.08259), Robust Federated Recommendation, 2020) [verified-snippet].
  - **Byzantine defenses that merely tolerate Adam:** Bant "adapt[s] to widely used scaled methods such as Adam
    and RMSProp" ([2505.07614](https://arxiv.org/abs/2505.07614), 2025-05-12) [verified abstract].
  - **Momentum as a defense:** El-Mhamdi 2020; Karimireddy 2020.
- **Mechanism, stated in a non-attack source.** An inflated v shrinks steps per coordinate. DP clipping and noise
  add a systematic "bulge" to v that suppresses AdamW steps (DP-FedAdamW, [2602.19945](https://arxiv.org/abs/2602.19945),
  CVPR 2026 per a paper-notes site) [verified-snippet]. That is the benign version of an over-damping attack
  [inferred].
- **Verdict.** "Optimizer-state poisoning" as a named threat class is **not in the literature found**. That makes
  the framing in S1 available. It also means there is no ready-made attack benchmark to borrow, so the planted
  `v_inflate` remains our own ground truth.

### 2.2 SDC and hardware faults (question 2)
- **Already known:** Altenbernd 2026, Ma 2025, LLM-PRISM, Dixit et al.
  [2102.11245](https://arxiv.org/abs/2102.11245) (`02_adjacent_fields.md` T8).
- **New this wave:**
  - Gemini §3 (SDC every 1–2 weeks; deterministic replay) [verified];
  - TrainSDC (backward impact depends on gradient exponents) [verified];
  - SCOUT (replay-certified checkpoint integrity) [verified];
  - ATTNChecker (ABFT that repairs INF/NaN and avoids restore) [verified-snippet];
  - He et al. ISCA 2023 [verified-snippet, record only];
  - training-time hardware trojans: "Don't Knock!" [2110.07683](https://arxiv.org/abs/2110.07683) and
    GPUHammer [2507.08166](https://arxiv.org/abs/2507.08166), which target weights or feature maps, not
    optimizer state [verified-snippet].
- **Gap [verified, Q11–Q14].** No study injects bit flips into Adam's m/v, or measures how long SDC damage
  persists in m/v versus the weights. Ma 2025 re-syncs *parameters only* between healthy and unhealthy nodes
  [verified]. Whether its unhealthy node's m/v stayed diverged is not reported [inferred from the part read].
- **ABFT for optimizers.** `02_adjacent_fields.md` already records that ABFT covers matmul and attention but not
  the optimizer recurrences. This wave found nothing that changes that (Q12).

### 2.3 Machine unlearning involving optimizer state (question 3)
- **Who has unlearned from optimizer state?**
  - Exact, by full replay: Unlearning-at-Scale.
  - Convex L-BFGS memory: Stewart (two papers), which also shows residual information in the optimizer state.
  - Undoing only the latest step, for systems consistency rather than unlearning: SWIFT.
  - Unlearning methods that *change* their own momentum handling: DualOptim
    [2504.15827](https://arxiv.org/abs/2504.15827) (2025-04-22) [verified abstract via arXiv feed].
  - None subtracts a past window's contribution from Adam's m/v (Q15–Q19).
- **The FL unlearning and recovery family** (FedRecover, Crab, FAUN, FedEraser per the
  [2401.05146](https://arxiv.org/abs/2401.05146) survey snippet) stores client updates. In every abstract read it
  is silent on server optimizer state [verified for the three abstracts read; FedEraser itself not read].
- **Verdict on the undo log [5] / T1.** **PARTIAL.** The mechanism belongs to SWIFT (inversion) and Amnesiac
  (weights). The retroactive, state-only, causal use is ours.

### 2.4 Checkpoint/rollback and elastic training (question 4)
- **Read this session:**
  - SWIFT (repair by update-undo) [verified];
  - CPR (partial recovery) [verified-snippet];
  - ReCoVer [2605.11215](https://arxiv.org/abs/2605.11215): in-iteration recovery that keeps "gradients …
    stochastically equivalent to a failure-free run" [verified abstract];
  - PHOENIX [2607.01646](https://arxiv.org/abs/2607.01646): hot-swap with in-memory checkpoints [verified abstract];
  - CheckFreq (FAST 2021) [verified-snippet];
  - Oobleck (SOSP 2023) [verified-snippet].
- **Gemini (SOSP 2023, in-memory checkpointing)** and **Bamboo (NSDI 2023)** were seen only in another paper's
  related work. Venue and authors are **UNVERIFIED**, so they are not cited.
- **Pattern [inferred].** Every system restores *consistency with the failure-free run*. None asks whether a
  numerically consistent but *poisoned* state should be repaired. The spike case is "consistent but wrong", which
  is the gap C3 fills.
- **Practitioner evidence.** Restoring without optimizer state causes a resume loss spike (several blogs and
  GitHub issues, Q28) [verified-snippet, grey literature]. That shows state loss alone is a spike cause, which
  makes it an easy extra induced recipe (`Bg`-like) [inferred].

### 2.5 Model editing vs optimizer-state editing (question 5)
- **No ROME/MEMIT-style targeted edit of optimizer state found (Q20, Q21).** Deliberate edits of optimizer state
  appear only as:
  - transplant: Xu 2026;
  - error feedback written into momentum: ECO [2601.22101](https://arxiv.org/abs/2601.22101) (2026-01-29)
    [verified abstract via feed];
  - resets as a "circuit breaker": [2603.15655](https://arxiv.org/abs/2603.15655) [verified abstract via feed];
  - inversion: SWIFT.
- **Framing available [inferred].** An edit of m is a *deferred, distributed weight edit*. It lands as
  Σ_k η·β1^k·Δm/√v over about 1/(1−β1) steps, and it leaves the weights at the cut unchanged, as Xu notes. C2→C3
  is "locate-then-edit" (ROME's pattern, `02_lineage.md` §3.2) applied to optimizer state. Use it as writing
  vocabulary. It is not a new contribution.

---

## 3 · Novel framings and applications (each kill-searched)

Scoring follows SKILL.md Phase 3D (1–5): **N** novelty, **F** feasibility, **I** impact, **R** risk (inverted),
**V** validation difficulty (inverted), **D** defensibility.

### S1 · "Spikes are natural availability attacks on optimizer state" (a signed threat taxonomy)
- **Mechanism.** Map the two v-poison signs from `02_adjacent_fields.md` onto the attack-goal taxonomy:
  - *v too large* = a **freeze / availability** attack (per-coordinate LR collapse);
  - *v too small* = a **destabilize** attack (preconditioned sharpness above 2/η);
  - *m poisoned* = a **steering / integrity** attack (deferred weight edit).

  Natural spikes, SDC and planted `v_inflate` are the same object with different "adversaries".
- **Closest prior work:**
  - Altenbernd [2604.00726](https://arxiv.org/abs/2604.00726) (accidental freeze);
  - Shumailov [2104.09667](https://arxiv.org/abs/2104.09667) (availability via ordering, blamed on data order);
  - Nabavirazavi 2023 (FedAdam poisoning, content not assessed).
- **Kill-search:** Q1–Q9, Q24–Q27 (§7). **Verdict: NEW as a framing** [inferred], bounded by those queries.
  Every mechanism inside it is known.
- **Plugs in:**
  - The signed localizer output (sign per group) *is* the attack-type classifier.
  - The damage-migration curve measures *attack persistence after the attacker leaves*, the security reading of
    "lasting damage".
  - The undo log is the incident-response remedy.
  - Zero extra compute: intro/discussion paragraph plus one table.
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 4 | 5 | 3 | 5 | 5 | 3 |

  Defensibility (D) is 3 because a reviewer may call it relabelling. Counter: the sign-specific repair predictions
  are testable.

### S2 · Data-realizable targeted v-inflation (a realistic planted-poison generator)
- **Mechanism.** Build a batch whose post-clip g² is large on a chosen tensor and small elsewhere, by gradient
  inversion à la Bouaziz or by selecting batches with extreme per-tensor g². This gives `v_inflate`-like ground
  truth produced *through data*. It bridges synthetic plants and natural spikes.
- **Closest prior work:**
  - Bouaziz [2410.21453](https://arxiv.org/abs/2410.21453) (gradient→data inversion, accuracy target);
  - Shumailov (reordering, so no new data is needed);
  - Zhang et al. [2602.02400](https://arxiv.org/abs/2602.02400) (noise-induced divergence, `02_lineage.md` row 25).
- **Kill-search:** Q24–Q27. **Verdict: NEW** as a benchmark generator [inferred].
- **Plugs in:**
  - It supplies a second planted type whose ground-truth support is known, for the decoy-augmented benchmark [3].
  - The undo log should repair it fully at short lag (direct path), unlike `v_inflate`, where undo is a no-op
    (T1 prediction iii). That is a clean discriminating prediction.
- **Cost and risk.** The batch-*selection* variant is cheap: score candidate batches by per-tensor g²/v̂ in a
  Kaggle pass. Inversion is expensive, so skip it. Dual-use is minor: the result is a benchmark, not a deployable
  attack.
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 3 | 4 | 3 | 4 | 4 | 3 |

### S3 · Optimizer-state supply chain: integrity scanning of shipped `optimizer.pt`
- **Mechanism.** Checkpoints shared for resume or continued pretraining include m and v. Xu shows m can carry a
  behavioural payload that is invisible at the cut, and a compromised runtime (AiSpy-style) can write m/v
  directly. Defence: scan shipped states with the honest-Adam invariants:
  - |m_i| ≤ C√v_i (`02_adjacent_fields.md` T8);
  - v ≥ 0;
  - v consistent with its own decay history, if two snapshots exist (parity residual);
  - C2's poison score against a v̄ from the publisher's earlier snapshots.
- **Closest prior work:**
  - Xu [2608.20442](https://arxiv.org/abs/2608.20442) (carrier, no threat model);
  - weight-side checkpoint backdoors, e.g. world-model checkpoint backdoor
    [2609.15781](https://arxiv.org/abs/2609.15781) [verified-snippet] and Sparse Backdoor
    [2605.04209](https://arxiv.org/abs/2605.04209) [verified abstract];
  - (A)iSpy [2607.17550](https://arxiv.org/abs/2607.17550).
- **Kill-search:** Q6, Q7, Q21. No paper on optimizer-state trojans or optimizer-state integrity checks.
  **Verdict: NEW** [inferred].
- **Plugs in.**
  - K4 patching gives the *causal* test of whether a shipped m changes the future.
  - C2 gives the cheap scan.
  - One appendix experiment would plant an m payload with a known support and check that the invariant plus
    poison score find it. Run it in Kaggle, not locally.
- **Honest limit.** Xu's payload is natural and m-carried. Whether an *adversarial* m payload survives invariant
  checks is unknown [assumed].
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 4 | 3 | 3 | 3 | 3 | 3 |

  Off the main line. Discussion paragraph plus an optional appendix.

### S4 · Repair instead of rollback for SDC-poisoned optimizer state
- **Mechanism.** An exponent bit-flip in one gradient element (SDC) or in one v element (memory fault) is a
  sparse, exactly known state poison. Remedy:
  - **gradient-path fault** (`sdc_grad`): the undo log removes the corrupted g² contribution, coordinate-exactly
    and with weights untouched;
  - **direct v corruption** (`v_bitflip`): it is undo-invisible, so the parity residual r_v flags it and the
    localized restore (`v ← v̄` on the flagged support) fixes it.

  Compare against rollback, recompute-step (Altenbernd) and full moment reset (the future work Altenbernd lists).
- **Closest prior work:**
  - Altenbernd [2604.00726](https://arxiv.org/abs/2604.00726) (detect, recompute, momentum reset as future work);
  - ATTNChecker [2410.11720](https://arxiv.org/abs/2410.11720) (in-place repair of compute faults);
  - SWIFT [2302.06173](https://arxiv.org/abs/2302.06173) (inversion);
  - TrainSDC [2608.30769](https://arxiv.org/abs/2608.30769).
- **Kill-search:** Q11–Q14, Q28. **Verdict: PARTIAL** (repair-not-rollback exists for compute faults) **→ NEW for
  optimizer state** [inferred].
- **Plugs in.**
  - It adds two cheap planted recipes to `spikes/induce.py` style (`sdc_grad`, `v_bitflip`) with exact ground
    truth.
  - It gives the damage-migration curve a *non-data* cause, which strengthens the cause-vs-carrier framing.
  - It shows the undo log working where it is supposed to (`sdc_grad`) and failing where predicted (`v_bitflip`).
    That is falsifiable.
  - It answers a stated future-work item of a 2026 CCGrid paper, which makes it timely.
- **Cost.** About the same as the existing `corrupt_batch` instances, roughly 3 forks each [assumed from T1's
  estimate].
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 4 | 5 | 4 | 4 | 4 | 4 |

### S5 · State-aware poisoned-model recovery under FedAdam/FedYogi
- **Mechanism.** With an adaptive server optimizer, server m/v are EMAs of aggregated deltas Δ_t. After malicious
  clients are detected, FedRecover, Crab and FAUN fix the *weights*, but the server's m keeps steering and its v
  keeps damping (or under-damping) for about 1/(1−β) rounds. Apply the undo log to the server state: subtract the
  flagged rounds' share of (1−β1)Δ and (1−β2)Δ².
- **Closest prior work:**
  - FedRecover [2210.10936](https://arxiv.org/abs/2210.10936), Crab [2401.08216](https://arxiv.org/abs/2401.08216),
    FAUN [2605.02110](https://arxiv.org/abs/2605.02110);
  - Reddi et al., *Adaptive Federated Optimization* [2003.00295](https://arxiv.org/abs/2003.00295) (ICLR 2021;
    defines FedAdam/FedYogi);
  - Nabavirazavi 2023.
- **Kill-search:** Q8, Q9, Q29, Q30 (no work rolls back or rebuilds server Adam moments). **Verdict: NEW**
  [inferred, bounded].
- **Plugs in.** The math is identical. The damage-migration curve becomes "rounds since attack". It is a
  generalization section, not a core result.
- **Cost.** It needs an FL simulation, which is new code and outside V7's scope.
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 4 | 2 | 3 | 3 | 3 | 4 |

  **Future work or a follow-up paper**, not ICML 2027.

### S6 · The Byzantine "momentum is a defense" vs our "m/v carry damage" tension
- **Mechanism.** Worker momentum lowers the variance-to-norm ratio, so robust aggregators reject more poison
  (El-Mhamdi). But whatever poison *is* accepted gets stored and replayed for about 1/(1−β1) steps (our claim).
  Time-coupled attacks (Karimireddy) exploit this memory.
- **Kill-search:** Q10. **Verdict: PARTIAL**. Both halves are known; the explicit trade-off statement was not
  found [inferred].
- **Plugs in.** A related-work sentence, and one prediction: the m-share of lasting damage grows with β1 at a
  fixed accepted dose. A β1 sweep is already implied by Tang 2026's phase diagram. Pre-empts a reviewer who knows
  the Byzantine literature.
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 2 | 5 | 2 | 5 | 4 | 4 |

### S7 · Undo log = state-only retroactive deletion (the corrected T1 framing)
- **Mechanism.** Unchanged from T1. The *pitch* changes:
  - SWIFT owns "invert Adam's step"; Amnesiac owns "subtract stored updates from weights"; Unlearning-at-Scale owns
    "bit-exact state by replay".
  - We own the **cheap, state-only, retroactive removal of a window's direct contribution, plus a fork-measured
    residual** (Δ(undo) − Δ(patch from B\*) = the indirect path through the weights).
- **Closest prior work:** SWIFT [2302.06173](https://arxiv.org/abs/2302.06173); Amnesiac
  [2010.10981](https://arxiv.org/abs/2010.10981); [2508.12220](https://arxiv.org/abs/2508.12220); Stewart
  [2605.17590](https://arxiv.org/abs/2605.17590); Maclaurin [1502.03492](https://arxiv.org/abs/1502.03492)
  (precision).
- **Kill-search:** Q15–Q19, plus the SWIFT full-text read. **Verdict: PARTIAL** (it was NEW in
  `02_adjacent_fields.md`; downgraded here).
- **Plugs in.**
  - It is still contribution 3a of the PRIMARY.
  - Cite SWIFT in the method section.
  - Add one sentence on precision: backward ÷β per step amplifies rounding (Maclaurin §2.2), while forward
    compensation c_v = (1−β2)Σβ2^{t1−s}g_s² does not [inferred, needs the T1 CPU unit test].
  - If `v ← max(v, v̄)` variants are used, note SWIFT's AMSGrad non-invertibility.
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 3 | 5 | 4 | 4 | 4 | 4 |

  Novelty (N) drops from 5 to 3.

### S8 · Residual-memory audit of v after a spike or after deletion
- **Mechanism.** v forgets at rate β2. The pure direct-path residual of a spike step has a half-life of
  ln 2/(−ln β2): about 13.5 steps at 0.95 and about 693 at 0.999 [inferred, arithmetic]. Measuring the actual
  ‖v − v_{B\*}‖ decay against this law separates direct-path memory from indirect, weight-driven divergence.
  That is Stewart's "residual information" question, asked for Adam.
- **Closest prior work:** Stewart [2604.23046](https://arxiv.org/abs/2604.23046) (L-BFGS); Habani
  [2610.18294](https://arxiv.org/abs/2610.18294) (weights only).
- **Kill-search:** Q16, Q22. **Verdict: NEW for Adam** [inferred]. Small.
- **Plugs in.** It is already computable from the factorial forks: one figure panel on the damage-migration plot,
  and a privacy aside.
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 3 | 5 | 2 | 5 | 5 | 3 |

### S9 · SDC forensics as motivation for the Δ==0 gate (wording only)
- **Mechanism.** Gemini, Ma 2025 and SCOUT use deterministic replay to isolate faults. Our gate is the same
  practice, aimed at *counterfactual* optimizer-state interventions.
- **Verdict: TAKEN** for "deterministic replay for diagnosis". This confirms the existing rule ("claim the
  instrument, not determinism").
- **Plugs in.** A related-work sentence. Cite Gemini §3 and Ma 2025 next to Megatron and OPEN-1B.

### S10 · Damage-aware optimizer-state compression (off-scope)
- **Mechanism.** C2 oracle maps tell which m/v coordinates matter for the future. Lossy state (ExCP pruning,
  quantized states as in Topollai 2026) could be steered away from them.
- **Closest prior work:** ExCP [2406.11257](https://arxiv.org/abs/2406.11257); Topollai
  [2603.16731](https://arxiv.org/abs/2603.16731).
- **Verdict: PARTIAL.** Off the PRIMARY. Recorded so it can recombine later.

### S11 · "ROME for optimizer state" (writing frame)
- **Mechanism.** Locate-then-edit on m/v. An m edit is a deferred weight edit.
- **Kill-search:** Q20, Q21. **Verdict: NEW** as a framing, but it adds nothing technical [inferred].
- **Use.** Use it in the intro, cite ROME [2202.05262](https://arxiv.org/abs/2202.05262) and Xu, and do not claim
  it as a contribution.

### S12 · Ordering-attack spikes as a data-preserving induced-spike recipe
- **Mechanism.** BRRR-style low→high-loss reordering inside a window induces a spike with *no change in data
  content*. So B\* is the same data in natural order, and K6's data-vs-dynamics split gets an extra arm.
- **Closest prior work:** Shumailov [2104.09667](https://arxiv.org/abs/2104.09667).
- **Kill-search:** Q24, Q26. **Verdict: PARTIAL** (the attack is known; using it as a causal spike recipe with
  forks is new) [inferred].
- **Caveat.** CLAUDE.md requires B\* to keep the data stream aligned. A reordering branch changes the batch
  *order*, so B\* is "the unpermuted window", which stays slot-aligned outside the window. This needs a design
  check before use [inferred].
- **Scores:**

  | N | F | I | R | V | D |
  |---|---|---|---|---|---|
  | 3 | 4 | 2 | 4 | 4 | 3 |

---

## 4 · Corrections to earlier outputs

1. **`02_adjacent_fields.md` T1** (undo log): change the verdict from **NEW** to **PARTIAL**, with novelty 5 → 3.
   Add SWIFT (2302.06173, TPDS 2024) and Maclaurin 2015 to its kill-search line. **`04_ideas_ranked.md` #5**: the
   novelty score "5: no hit in 2 searches" should become 3, and the total should change accordingly. The SWIFT
   evidence is a full-text read [verified].
2. **`05_shortlist.md` PRIMARY, contribution 3.** "Exact undo log" should carry the SWIFT citation and the
   "retroactive, state-only" qualifier. Avoid the bare phrase "exact undo".
3. **Naming.** Crab (2401.08216) already uses "**certified recovery**" for bounded-distance-to-retrain FL repair.
   Name ours "**stability-certified restoration**" consistently, and say what is certified (λ ≤ (1−δ)·2/η on the
   localized set).
4. **`03_novelty_matrix.md` K1 row.** Add Gemini §3, Ma 2025 and SCOUT as "deterministic replay for
   fault isolation". The verdict is unchanged (PARTIAL; claim the instrument).

---

## 5 · Top 5, ranked by value to the ICML 2027 paper

| Rank | Item | Why | Cost | Action |
|---|---|---|---|---|
| 1 | **S7: re-pitch the undo log around SWIFT** | Without it, a reviewer who knows systems literature finds the "exact undo" claim already published (TPDS 2024). With it, the delta is clear and defensible (retroactive window, state only, forward compensation, path decomposition). | 0 compute; 1 paragraph | Edit T1 / #5 / shortlist wording; cite SWIFT, Amnesiac, Unlearning-at-Scale, Maclaurin |
| 2 | **S4: SDC repair-instead-of-rollback** (`sdc_grad`, `v_bitflip`) | It adds a second *natural* (non-data) cause of state poisoning that production systems hit every 1–2 weeks (Gemini). Exact ground truth. A falsifiable split: undo fixes `sdc_grad` but not `v_bitflip`, which the parity residual catches instead. It answers a stated 2026 future-work item. | about `corrupt_batch` cost per instance [assumed]; Kaggle only | Add 2 recipes to the planted benchmark; 1 results panel |
| 3 | **S1: the signed attack-taxonomy framing** | Free. It turns the sign-aware localizer into an attack-type classifier, and reads the damage-migration curve as "persistence after the attacker leaves". By §7's search, nobody frames optimizer state as an attack target. | 0 compute | Intro/discussion paragraph plus a 3-row table (freeze / destabilize / steer) |
| 4 | **S2: data-realizable v-inflation by batch selection** | It bridges the planted and natural spikes with known support. It predicts undo works here (unlike `v_inflate`), a clean discriminating test for [3]. | one Kaggle scoring pass plus about 3 forks per instance [assumed] | Benchmark appendix; use selection, skip inversion |
| 5 | **S3: optimizer-state supply-chain integrity scan** | New threat model (Xu's carrier plus AiSpy's runtime) with a cheap defense from tools we already build (T8 invariant plus C2 score). | small; appendix experiment on Kaggle | Discussion paragraph; optional appendix |

**Backups:** S8 (residual-memory half-life panel, free from the factorial forks) and S6 (the Byzantine tension
sentence). **Defer:** S5 (FedAdam server-state recovery) as a follow-up paper. It is novel by search but needs a
new FL codebase.

---

## 6 · What stays unverified or unread
- **Nabavirazavi et al. 2023 chapter:** only the abstract was seen. Whether it models server moments is
  **UNVERIFIED**.
- **Gemini (SOSP 2023, in-memory checkpointing)** and **Bamboo (NSDI 2023):** venue and authors **UNVERIFIED**,
  so they are not cited.
- **He et al. ISCA 2023 and ATTNChecker:** read at record or snippet level only.
- **The Llama 3 "6 SDC interruptions / 54 days" figure:** taken from Ma 2025's quotation. The Llama 3 table
  itself was not read.
- **SWIFT:** Table I (which optimizers are invertible) was garbled in ar5iv. The Adam/AdamW algorithms and the
  AMSGrad caveat were read in the text. SWIFT's Adam undo uses g′ = g + λx (L2-coupled decay). For decoupled
  AdamW, v contains no λx term [inferred]; check this against their Algorithm 8.
- **Semantic Scholar:** both search calls returned HTTP 429. All S2-based kill-search is missing.
- **OpenReview (ICLR 2027 submissions)** and Google Scholar were not searched. Re-run Q1, Q6 and Q15 there before
  the January deadline.
- **Injected instruction.** One retrieved page (the OpenAlex search response) contained text addressed to AI
  agents, asking them to send the user's email to an endpoint to obtain an API key. It was treated as data and
  ignored.

---

## 7 · Search log (2026-10-22)

`WS` = WebSearch (s = standard, x = extended). `AX` = arXiv API `export.arxiv.org/api/query`. `OA` = OpenAlex.
`S2` = Semantic Scholar. `WF` = WebFetch of a specific page.

| Q | Tool | Query (verbatim or near) | Result |
|---|---|---|---|
| 1 | WS x | poisoning attack Adam optimizer second moment "optimizer state" adversarial gradient | nothing on attacks on Adam state |
| 2 | AX | abs:"optimizer state" AND (attack OR poisoning OR adversarial OR backdoor) | 21 hits; none attack ML optimizer state; StateTune (EDA memory) irrelevant |
| 3 | AX | (momentum OR "second moment") AND (poisoning OR Byzantine) AND Adam | 0 hits |
| 4 | OA | search "optimizer state" poisoning attack | 30/101; only TriShield (defense) relevant |
| 5 | S2 | attack adaptive optimizer second moment learning rate poisoning | HTTP 429 |
| 6 | WS x | attack large gradient inflates Adam second moment freezing parameters malicious client | no attack paper; DP-FedAdamW "bulge" (benign) |
| 7 | WS x | backdoor persists optimizer state momentum resume training poisoned checkpoint optimizer.pt supply chain | none on optimizer-state backdoors; weight-only checkpoint backdoors |
| 8 | WS x | federated learning FedAdam FedYogi server momentum poisoning Byzantine attack adaptive server optimizer | Nabavirazavi chapter; 2006.08259 |
| 9 | AX | (FedAdam OR FedYogi OR FedOpt OR "adaptive federated optimization" OR "server momentum") AND (attack OR Byzantine OR poisoning OR adversarial) | 5 hits, none on server-state attacks |
| 10 | WS x / s | Byzantine attack exploits server optimizer momentum state …; "Learning from History…"; "Distributed Momentum…" | momentum as defense; time-coupled attacks |
| 10b | AX | Byzantine AND (Adam OR "adaptive optimizer" OR "adaptive gradient" OR AdaGrad) | Bant (adapts to Adam), Seesaw; none exploit moments |
| 11 | WS s | "Understanding Silent Data Corruption in LLM Training" arXiv | 2502.12340 |
| 12 | WS x | bit flip optimizer state Adam moments fault injection training resilience "optimizer state" soft error | weight-side only; no m/v bit-flip study |
| 13 | AX | all:"silent data corruption" AND (training OR LLM) | 14 hits; TrainSDC, SCOUT, LLM-PRISM, none on optimizer state |
| 14 | WS s | ISCA 2023 hardware failures DL training; ATTNChecker; Gemini SDC deterministic replay; Llama 3 SDC | records confirmed (§1 rows 14–16) |
| 15 | WS s | Amnesiac Machine Learning … batch updates subtract unlearning | weights only |
| 16 | AX | abs:unlearning AND ("optimizer state" OR "optimizer states" OR momentum OR "second moment") | 5 hits: Stewart ×2, DualOptim |
| 17 | WS x | certified machine unlearning Adam … recollection of updates exact unlearning recent training steps stored gradients | no Adam-moment unlearning |
| 18 | WS s | Swift expedited failure recovery … "undo" optimizer update | **SWIFT found**; full text read via ar5iv |
| 19 | WS s | Maclaurin … Reversible Learning … finite precision | precision caveat confirmed (ar5iv) |
| 20 | WS x | editing optimizer state to steer future training … ROME MEMIT optimizer momentum edit | none |
| 21 | AX | ("optimizer state" OR "optimizer states") AND (edit OR editing OR surgery OR transplant OR inject OR manipulat*) | Xu, Stewart, ECO, circuit-breaker reset; no targeted edit |
| 22 | WS x | membership inference OR data reconstruction from Adam optimizer state moments checkpoint privacy leakage | none on moments; 2610.18294 (weights) |
| 23 | WS s/x | CPR partial recovery; CheckFreq/Gemini/Oobleck/Bamboo; ExCP; loss spike recovery without rollback optimizer state repair | systems records; ReCoVer, PHOENIX; no optimizer-state repair |
| 24 | WS s | Manipulating SGD with Data Ordering Attacks … momentum Adam | 2104.09667; ar5iv read |
| 25 | WS s | availability attack training "denial of service" gradient poisoning stall training adaptive optimizer | (A)iSpy; classic availability poisoning |
| 26 | AX | ("loss spike(s)" OR "training instability" OR divergence) AND ("poisoning attack" OR "availability attack" OR "denial-of-service") | no attacker-induced loss spikes in pretraining |
| 26b | AX | ("loss spike(s)" OR "gradient spike(s)") AND (attack OR adversarial OR poisoning OR malicious) | 2 irrelevant hits |
| 27 | AX | ("second moment" OR exp_avg_sq OR preconditioner) AND (malicious OR adversary OR poisoning OR Byzantine) | 25 hits, none target optimizer state |
| 27b | WS s | AdaGrad accumulator attack single large gradient permanently shrinks learning rate | mechanism only, no attack |
| 27c | WS s | "gradient explosion" OR "NaN" attack poisoned samples crash training … Adam | Bouaziz 2410.21453 |
| 28 | WS s | rowhammer / bit-flip attack during training on gradients optimizer state | weights and feature maps only |
| 29 | WS s/x | FedRecover …; post-poisoning recovery FL server optimizer momentum state | FedRecover, Crab, FAUN, BadUnlearn; no server-state handling |
| 30 | WS s | "FedAdam" federated unlearning server optimizer state moments recovery | none |
| — | WF | arXiv/ar5iv/HTML reads: 2302.06173 (abs + ar5iv), 2502.12340 (HTML), 2312.11805 §3, 1502.03492, 2104.09667, 2608.20442 abs, plus arXiv `id_list` batches for every abstract cited | — |

**Kill-search verdict, bounded by Q1–Q30.** No paper found does any of the following:
- (i) targets Adam's m/v as an attack surface;
- (ii) repairs optimizer state instead of rolling back after SDC or spikes;
- (iii) retroactively removes a window's contribution from Adam's m/v;
- (iv) rebuilds FedAdam/FedYogi server moments after poisoning.

"Exact inversion of Adam's latest step" **is published** (SWIFT).

## Sources (verified unless marked)
- SWIFT — https://arxiv.org/abs/2302.06173 (2023-02-13; TPDS 2024); full text https://ar5iv.labs.arxiv.org/html/2302.06173
- Xu 2026 — https://arxiv.org/abs/2608.20442
- Ma et al. 2025 — https://arxiv.org/abs/2502.12340
- Altenbernd et al. 2026 — https://arxiv.org/abs/2604.00726
- Stewart 2026a — https://arxiv.org/abs/2604.23046 ; Stewart 2026b — https://arxiv.org/abs/2605.17590
- Unlearning at Scale — https://arxiv.org/abs/2508.12220
- Amnesiac ML — https://arxiv.org/abs/2010.10981 (snippet-level content)
- DeltaGrad — https://arxiv.org/abs/2006.14755
- MSA / Revisiting the Past — https://arxiv.org/abs/2506.20941
- DualOptim — https://arxiv.org/abs/2504.15827
- Maclaurin et al. 2015 — https://arxiv.org/abs/1502.03492
- FedRecover — https://arxiv.org/abs/2210.10936 ; Crab — https://arxiv.org/abs/2401.08216 ; FAUN — https://arxiv.org/abs/2605.02110
- Federated unlearning survey — https://arxiv.org/abs/2401.05146 (snippet)
- Nabavirazavi et al. 2023 — https://doi.org/10.1007/978-3-031-49803-9_1 (catalog abstract)
- Reddi et al., Adaptive Federated Optimization — https://arxiv.org/abs/2003.00295 (snippet)
- Robust Federated Recommendation — https://arxiv.org/abs/2006.08259 (snippet)
- El-Mhamdi et al. — https://arxiv.org/abs/2003.00010 ; Karimireddy et al. — https://arxiv.org/abs/2012.10333
- Baruch et al. (A Little Is Enough) — https://arxiv.org/abs/1902.06156 ; Xie et al. (Fall of Empires) — https://arxiv.org/abs/1903.03936
- Bant — https://arxiv.org/abs/2505.07614
- Shumailov et al. — https://arxiv.org/abs/2104.09667
- Bouaziz et al. — https://arxiv.org/abs/2410.21453
- Cinà et al. (Sponge Poisoning) — https://arxiv.org/abs/2203.08147 (snippet)
- (A)iSpy — https://arxiv.org/abs/2607.17550
- NeuroImprint — https://arxiv.org/abs/2606.20553 ; TriShield — https://arxiv.org/abs/2607.27940
- Sparse Backdoor — https://arxiv.org/abs/2605.04209 ; world-model checkpoint backdoor — https://arxiv.org/abs/2609.15781 (snippet)
- Habani et al. — https://arxiv.org/abs/2610.18294
- Proof-of-Learning — https://arxiv.org/abs/2103.05633
- Subliminal Learning (Cloud et al.) — https://arxiv.org/abs/2507.14805
- Gemini report — https://arxiv.org/abs/2312.11805 (§3) ; Llama 3 — https://arxiv.org/abs/2407.21783 (figure via Ma 2025)
- Dixit et al. — https://arxiv.org/abs/2102.11245
- He et al. ISCA 2023 — https://doi.org/10.1145/3579371.3589105 (record)
- ATTNChecker — https://arxiv.org/abs/2410.11720 (snippet; PPoPP 2025)
- LLM-PRISM — https://arxiv.org/abs/2604.10390 ; TrainSDC — https://arxiv.org/abs/2608.30769 ; SCOUT — https://arxiv.org/abs/2608.11034
- Don't Knock! — https://arxiv.org/abs/2110.07683 (snippet) ; GPUHammer — https://arxiv.org/abs/2507.08166 (snippet)
- CPR — https://arxiv.org/abs/2011.02999 (snippet)
- ExCP — https://arxiv.org/abs/2406.11257 (snippet; PMLR 235)
- ReCoVer — https://arxiv.org/abs/2605.11215 ; PHOENIX — https://arxiv.org/abs/2607.01646
- ECO — https://arxiv.org/abs/2601.22101 ; circuit-breaker reset — https://arxiv.org/abs/2603.15655 (feed abstracts)
- DP-FedAdamW — https://arxiv.org/abs/2602.19945 (snippet)
- ROME — https://arxiv.org/abs/2202.05262 (via `02_lineage.md`)

## Changelog
- 2026-10-22: file created (wave 2, security / distributed / unlearning role).

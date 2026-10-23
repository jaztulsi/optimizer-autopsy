# 12 — Citation fact-check for PLAN_V8 (C1–C25)

Verifier pass, 2026-10-22. Each claim was checked against a primary source: an arXiv abs/HTML page, the ACL Anthology, PMLR, the NeurIPS/ICML virtual site, an HF repo tree or model card, the GCS bucket listing, or an official blog. Quotes are ≤25 words. "Scope" means the number is real but belongs to a narrower setting than the plan sentence implies.

Verdict counts (C16's four papers count as one claim): **CONFIRMED 20** (C6, C8, C10, C11, C12 and C22 carry scope/wording/venue notes) · **CORRECTED 5** (C5, C13, C17, C21, C24) · NOT FOUND 0 · UNREACHABLE 0. C2's full PDF was unreachable (an OpenReview browser challenge), but its verbatim abstract was verified on neurips.cc.

## Corrections that must propagate into PLAN_V8

1. **C5 (Liu & Li 2608.19762)**: the "responses peak near horizons 6/16/26" result comes from the **controlled quadratic systems** in §4.1, under exact θ/m/v state interventions. It is not from Pythia. The Pythia 410M/1B/1.4B runs use H = 8, are "a single-seed scaling analysis" with descriptive statistics, and test θ/m/v channels through "eight post-shock hybrid states" (App. E.2.3). The main shock is one gradient perturbation entering all three blocks, not a per-channel injection. The plan should say: *"paired control/shock trajectories (§3.2); on quadratic systems, isolated θ/m/v responses peak near horizons 6/16/26 (§4.1); on Pythia-410M/1B/1.4B, continued 500 AdamW updates, hybrid-state channel tests at H = 8 (App. E.2.3)."*
2. **C13 (Bai et al. 2506.04805)**: the paper never writes "38/η". Prop. 4.2 gives λ_i < (2/η)(1+β1)/(1−β1). That equals 38/η at β1 = 0.9, but the arithmetic is ours. Eq. 7 is the *definition* of Ĥ, not that threshold. Cite 38/η to **Cohen et al. 2207.14484 (abstract)** instead, or write "Prop. 4.2, evaluated at β1 = 0.9". Code: **no public repo**. App. B says code is "in the supplementary materials", and the arXiv, PMLR and ICML pages link none.
3. **C17 (Andersen–Bollerslev–Diebold 2007)**: "far less persistent" is wrong; the published abstract says "**distinctly less persistent** than the continuous component". "Contribute little to forecasts" comes from the **2005 NBER WP 11775** abstract ("almost all of the predictability … comes from the non-jump component"). The published abstract says instead that separating jumps gives "significant out-of-sample volatility forecast improvements". Cite the WP for the predictability line, or reword.
4. **C21 (Ma et al. ACL 2025)**: the paper does **not** use parameter-only synchronization. It uses **computation synchronization** (RQ1, submodule outputs and gradients overwritten), **parameter synchronization** (RQ2, single step) and **no synchronization** (RQ3, drift). "Does not discuss optimizer state" is CONFIRMED: the phrase never appears.
5. **C24 (Marin 32B retro)**: the spikes were **not traced** to attention. The retro *infers* by elimination that the instability was "rooted in the model's attention stack rather than in optimizer state". QK-Norm (via the switch to a Qwen3-32B backbone) made spikes "disappear entirely". **This cuts against our thesis.** A "necromancy" optimizer-state rebuild "stabilized gradients for a few thousand steps, but the run relapsed". PLAN_V8 should cite it as a counter-case, not as support.
6. **C10 (LLM-PRISM)**: the abstract says the faults "perturb gradients and parameters". The actual injection sites are forward activations and backward input/weight gradients (§III-C1c), so parameters are affected only indirectly. Say "injects into activations and gradients; never into optimizer state". The faults are *permanent GPU faults* (RTL-level simulation), on GPT-2 Small and Medium.
7. **Scope qualifiers (numbers correct, setting narrower):**
   - **C6**: 89.9% is one Qwen isolated-pulse cell at lag 24, measured as an adjoint allocation of the signed source contribution (θ 10.6%, v −0.5%). Say "in one Qwen cell".
   - **C8**: C ≈ 2.21, p ≈ 0.92 is the modular-division Transformer fit (Fig. 1b). Across the six models, p = 0.75–1.05, and the abstract describes the boundary as "approximately linear".
   - **C11**: the paper never says the moments are released. The HF model card does.
   - **C22**: ZClip has no venue. The ICML 2026 venues for AdaGC and GradientStabilizer rest on the authors' arXiv comments; no PMLR v306 page was located.
   - **C15**: AFOR is an EEG-decoding paper, not an LLM paper.
   - **C19**: Crab is about federated-learning poisoning.

## High-value asset found (new, act on this)

**OPEN-1B publishes full optimizer state every 100 steps, publicly.** The HF card for `Gensyn/open-1b-base` says: "Checkpoint trajectory (every 100 steps, full optimizer state): `gs://gensyn-open-1b/ckpt`". The bucket lists anonymously. `ckpt/step_000000100/` holds about 19.31 GB of DCP shards, which is about 3× the 6.43 GB fp32 `model.safetensors` and consistent with w + m + v in fp32. It also holds per-rank RNG, `state_hash.txt` and `spike_protocol.json`, a skip-step log that was empty at step 100. `meta.json` gives `adamw_repop`, betas [0.9, 0.95], eps 1e-8, wd 0.1. The final step is 80,957. The shard contents were not opened, since no code was run. That makes this a natural-spike candidate with real m/v trajectories, alongside K2.

## Verdict table

| ID | Claim (short) | Verdict | Primary source (title · first author · year · venue · arXiv) | Locator / quote | Correction / note |
|---|---|---|---|---|---|
| C1 | SWIFT update-undo; venue TPDS 35(9) 2024 | CONFIRMED (venue) | *SWIFT: Expedited Failure Recovery for Large-scale DNN Training* · Yuchen Zhong · 2023/2024 · IEEE TPDS 35(9), Sep 2024, DOI 10.1109/TPDS.2024.3429625 · 2302.06173 | arXiv journal-ref field | Only the venue was re-checked; the update-undo mechanics rely on the earlier ar5iv check. |
| C2 | v-excess on recently-written rows protects those writes | CONFIRMED (abstract level) | *Recent Training is Protected by Adam's Second Moment* · Shrey Biswas (sole) · 2026 · NeurIPS 2026 Workshop OPT 2026 (poster, neurips.cc/virtual/2026/161497; OpenReview VJFfO4tLMB) · no arXiv | Abstract: "That excess later blocks new writes from permanently overwriting the older ones … until the protection decays." | **Setting: toy task** "with two equally valid solutions, so ordinary data favours neither"; two opposing "lessons". **Gap result:** 300-step wait → drifts back (first-lesson preference 0.40–0.97); 3,000-step wait → overwrite (0.01–0.10). **Decay:** protection decays "within 8% of the excess second moment's −1/ln β2" timescale. **Law:** the second lesson sticks iff "the ratio of the two lessons' integrated writes exceed[s] 0.63". **Causal test:** clearing excess v localised to the rows the second lesson writes → protection vanishes; "as effective as fully resetting optimiser state, and more effective than lowering β2", which "backfires as predicted by our law". **Claim:** the excess is "readable from two optimiser-state snapshots, editable in place". Full PDF unreachable (OpenReview challenge), so the architecture, β values and law equation are unverified. No LLM-scale evidence in the abstract. |
| C3 | neox-ckpt-pythia-160m-deduped-v1 has final ZeRO optimizer shards; hyperparams; email for intermediates | CONFIRMED | HF repo `EleutherAI/neox-ckpt-pythia-160m-deduped-v1`; GitHub `EleutherAI/pythia` README | Tree: `zero_pp_rank_{0..31}_mp_rank_00_optim_states.pt`, 32 × 60.9 MB, at the repo root. `160M-deduped.yml`: Adam, lr 6e-4, betas [0.9, 0.95], eps 1e-08, num-layers 12, hidden 768, heads 12, train-iters 143000. README: "final-step checkpoints+optimizer states downloadable"; other steps: "please email" the maintainers | none |
| C4 | PolyPythias: spikes only in 410M seeds 3 and 4 of 50 runs (14M–410M); optimizer states? | CONFIRMED; **optimizer states NOT released** | *PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs* · Oskar van der Wal · 2025 · ICLR 2025 · 2503.09543 | §3: "only for these very seeds we observe 'loss spikes'"; 45 new and 5 original runs, 14M–410M | "Only" rests on the outlier-seed heuristic (§3), not an exhaustive spike scan. Release covers ~7k checkpoints and pre-shuffled data; the HF `pythia-410m-seed3` tree has only `pytorch_model.bin`. |
| C5 | Liu & Li: paired forks; θ/m/v injection; peaks 6/16/26; Pythia 410M/1B/1.4B, 500 updates | **CORRECTED** (setting attribution) | *Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW* · Kang Liu · 2026 · arXiv preprint · 2608.19762 | §3.2 "a control trajectory and a shock trajectory initialized from the same state"; §4.1 peaks 6/16/26 (quadratic); §4.1 Pythia "is continued for 500 AdamW updates"; Table 3; App. E.2.3 | See correction 1. |
| C6 | Xu: m ≈ 90% (89.9%), v ≈ 0 (−0.5%); m-only transplant leaves params unchanged at cut | CONFIRMED (scope) | *Stored in Optimizer State, Valued by Later Training: A Causal Account of Subliminal Trait Transfer* · Qinyang Xu · 2026 · arXiv preprint · 2608.20442 | §3: "places 89.9% of the signed source contribution in the first moment"; Fig. 2(a); abstract: "Transplanting the first moment alone leaves the parameters, hidden states, and outputs unchanged at the cut." | One Qwen cell at lag 24; parameters diverge after the cut (Fig. 3c). Replicated on Qwen, SmolLM2, Llama and MNIST. |
| C7 | Xie et al.: rank-3 QK spectral projection with energy-matched control | CONFIRMED | *One QK Channel, Many Sources: Tracing Low-Precision Attention Collapse* · Shuxiao Xie · 2026 · arXiv preprint (v2 retitled) · 2608.02091 | Abstract: "projecting each update off the current QK weights' leading three singular directions" → σ_max 11.1, vs "removing equal energy elsewhere" → 237 | bf16 streaming-softmax fault; QK-Guard controller. |
| C8 | Tang/Chen/Liu: 1−β2 = C(1−β1)^p, C ≈ 2.21, p ≈ 0.92 | CONFIRMED (scope) | *Beyond Quadratic Loss: The Stability Phase Diagram of Adam* · Gaoxiang Tang · 2026 · arXiv preprint · 2609.18314 | Fig. 1(b) caption: boundary "follows the near-linear relation" 1−β2 ≃ 2.21(1−β1)^0.92; §2: "p=0.92,0.84,0.75,0.95,1.05,0.95" | The fit is for the modular-division Transformer only. Quadratic toy: γ ≃ 15.6δ^3.07. |
| C9 | Altenbernd et al.: SDC → v→∞ / freezes params; momentum resets as future work | CONFIRMED | *Exploring Silent Data Corruption as a Reliability Challenge in LLM Training* · Anton Altenbernd · 2026 · CCGrid 2026 · 2604.00726 | §IV-D Fig. 2: "causes the optimizer's second moment to become infinity, freezing parameter updates"; §VIII: "Future work should also explore proactive mitigation methods, such as momentum resets." | LLaMA 60M/350M/1.3B. |
| C10 | LLM-PRISM: injection into activations/gradients, not optimizer state; 7,664 Megatron runs | CONFIRMED (wording note) | *LLM-PRISM: Characterizing Silent Data Corruption from Permanent GPU Faults in LLM Training* · Abhishek Tyagi · 2026 · SC (arXiv comment; year not stated) · 2604.10390 | §III-C1c: "forward-pass activations (fwd_outputs)", "bwd_grad_inputs", "bwd_grad_weights"; §IV-C: 7,664 runs = 4,681 GPT-2 Small + 2,983 GPT-2 Medium; Megatron-LM | See correction 6. §V treats momentum only as a smoother. |
| C11 | OPEN-1B: AdamW (0.9, 0.95, 1e-8); ckpt every 100 steps; FP32 moments; exact replay; public moments? | CONFIRMED; **moments publicly released** (via the model card, not the paper) | *OPEN-1B: A Fully Auditable Training Run* · John Donaghy · 2026 · arXiv preprint · 2609.17380 | §3.2: "We use AdamW … with β1=0.9, β2=0.95, and ε=10−8"; "Optimizer state is checkpointed and hashed in FP32"; "an audit replay can reconstruct the moments exactly"; §1: checkpoints "at intervals of 100 steps". HF card: "every 100 steps, full optimizer state: gs://gensyn-open-1b/ckpt" | See "High-value asset" above. Code: github.com/gensyn-ai/open-transformers. |
| C12 | NVIDIA Megatron Core bitwise-deterministic blog (~Oct 2026) | CONFIRMED | *Scale Bitwise-Deterministic Pretraining with NVIDIA Megatron Core* · Ashwath Aithal · Oct 06, 2026 · NVIDIA Technical Blog | "two runs launched from the same initial state must remain bitwise identical at every training step"; resume must match "parameters, optimizer state, RNG state, data position"; "bitwise determinism over 800 steps" at 2,432 GPUs; ~2% overhead | Adam moments are not named, only "optimizer state". Valid only "within the same hardware and software environment". |
| C13 | Bai et al.: Ĥ with (1−β1)/(1+β1) and bias correction, threshold 2/η; 38/η at β1 = 0.9 (Prop 4.2 / eq 7); 187M LLaMA on 100B tokens; λ_grad; code? | **CORRECTED** (38/η locator; code) | *Adaptive Preconditioners Trigger Loss Spikes in Adam* · Zhiwei Bai · 2026 · ICML 2026, PMLR 306:5566–5601 (bai26o) · 2506.04805 | Eq. 7 §4.1 (Ĥ); "λmax(Ĥ)<2/η" after Eq. 5; Prop. 4.2: λ_i < (2/η)(1+β1)/(1−β1); §6.2 and App. G: "LLaMA structure Transformer with 187M non-embedding parameters", 100B SlimPajama tokens; Eq. 8 §4.3: λ_grad = gᵀHg/‖g‖² > 2/η predicts spikes | See correction 2. |
| C14 | Wortsman: 28/30 loss spikes follow an RMS spike by 1–8 iters (§3.4); App. D ViT-Huge 14/15 | CONFIRMED | *Stable and low-precision training for large-scale vision-language models* · Mitchell Wortsman · 2023 · arXiv page lists no venue · 2304.13013 | §3.4: "28 out of 30 detected loss spikes follow an RMS spike by 1-8 iterations"; App. D (Fig. 16): "14 out of 15 come 1-8 iterations after an RMS spike in the patch embedding layer" | Venue not verified on a primary page; cite as arXiv. |
| C15 | Kourkoutas-β and AFOR: adaptive β2 | CONFIRMED | *Kourkoutas-Beta: A Sunspike-Driven Adam Optimizer with Desert Flair* · Stavros C. Kassinos · 2025 · arXiv · 2508.12996 — *Adaptive Forgetting for Nonstationary Optimization: Towards Robust EEG Decoding* · Hongyu Zhu · 2026 · arXiv · 2609.24233 | Kourkoutas: "fixed second-moment discount beta2 is replaced by a layer-wise dynamic value", driven by the "sunspike" ratio of the current gradient norm to its EMA. AFOR: tensor-wise; "converts the fixed second-moment decay coefficient into a dynamic coefficient", driven by the RASS signal-quality score | AFOR's domain is EEG. |
| C16a | Guo 2608.24593 | CONFIRMED | *Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions* · Jinhui Guo · 2026 · arXiv | Full transport "lowers token-disjoint loss relative to an optimizer-aware immediate derivative in 10/12 histories" | The mean benefit is small. |
| C16b | Kim 2608.15824 | CONFIRMED | *Second-Moment Memory in Coordinatewise Adam* · Jeonseong Kim · 2026 · arXiv | "Long second-moment memory can slow optimization even when the gradient noise has finite variance" (smooth-convex lower bound) | none |
| C16c | "Hidden Ratio" NeurIPS 2026 poster 154051 | CONFIRMED | *The Hidden Ratio in Adam: Stable Structure, Compression, and Sign Dynamics* · Yihe Zhou · 2026 · NeurIPS 2026 poster (OpenReview GZha6judfp) | "adaptive dynamics can be expressed through a transformed ratio with approximately scale-stable behavior" (tied-β regime; 4-bit state; link to Signum) | Track not stated on the page. |
| C16d | Topollai & Choromanska 2603.16731 | CONFIRMED | *Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets* · Kristi Topollai · 2026 · arXiv | "once a quantized EMA becomes effectively stale, resetting it can temporarily restore responsiveness" | none |
| C17 | ABD 2007: jumps far less persistent; contribute little to forecasts | **CORRECTED** (wording) | *Roughing It Up: Including Jump Components in the Measurement, Modeling, and Forecasting of Return Volatility* · Torben G. Andersen · 2007 · Rev. Econ. Stat. 89(4):701–720, DOI 10.1162/rest.89.4.701 (WP: NBER w11775, 2005) | Published abstract (Duke Scholars): "distinctly less persistent than the continuous component"; WP abstract: "almost all of the predictability … comes from the non-jump component" | See correction 3. The MIT Press page returned 403; the published abstract was read via Duke Scholars. |
| C18 | Gemini §3: SDC "every week or two"; deterministic replay isolates bad computations | CONFIRMED | *Gemini: A Family of Highly Capable Multimodal Models* · Gemini Team, Google · 2023 · arXiv · 2312.11805 | §3: "we can expect SDC events to impact training every week or two"; "techniques that exploit deterministic replay to isolate incorrect computations" | none |
| C19 | Crab uses "certified recovery" | CONFIRMED | *Towards Efficient and Certified Recovery from Poisoning Attacks in Federated Learning* · Yu Jiang · 2024 · arXiv · 2401.08216 | Abstract: "we introduce Crab, an efficient and certified recovery method" | The setting is federated-learning poisoning. |
| C20 | Marek: β2 half-life fixed in tokens, β2* = β2^(B*/B) | CONFIRMED | *Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful* · Martin Marek · 2025 · arXiv (no venue on page) · 2507.07101 | Abstract: "hold its half-life fixed in terms of tokens"; §4.3 Eq. 2: β2* = β2^(B*/B) | none |
| C21 | Ma et al. ACL 2025: parameter-only sync; no optimizer-state discussion | **CORRECTED** | *Understanding Silent Data Corruption in LLM Training* · Jeffrey Jian Ma · 2025 · ACL 2025 (Long), pp. 20372–20394, DOI 10.18653/v1/2025.acl-long.996 · 2502.12340 | §1: "a computation synchronization mechanism"; §3.2: "broadcast the updated model parameters from the healthy node"; §6.1: no sync | See correction 4. Adam β2 = 0.99 (App. B.3). |
| C22 | ZClip, AdaGC (ICML 2026?), GradientStabilizer (ICML 2026?), SPAM (ICLR 2025) | CONFIRMED (venue notes) | *ZClip: Adaptive Spike Mitigation for LLM Pre-Training* · Abhay Kumar · 2025 · arXiv only · 2504.02507 — *AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping* · Guoxia Wang · 2025/26 · ICML 2026 ("Accept by ICML 2026", arXiv comment) · 2502.11034 — *GradientStabilizer: Fix the Norm, Not the Gradient* · Tianjin Huang · 2025/26 · ICML 2026 ("Accepted By ICML2026", arXiv comment) · 2502.17055 — *SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training* · Tianjin Huang · 2025 · ICLR 2025 (proceedings PDF; iclr.cc poster 30015) · 2501.06842 | arXiv comment fields; ICLR proceedings | ZClip has no venue. GradientStabilizer v1 was titled *Stable-SPAM: How to Train in 4-Bit More Stably than 16-Bit Adam*. AdaGC v1 was titled *AdaGC: Improving Training Stability for Large Language Model Pretraining*. No PMLR v306 page was found for AdaGC or GradientStabilizer. |
| C23 | Unlearning-at-Scale: bit-exact model + optimizer state via replay; Pythia/Llama | CONFIRMED | *Unlearning at Scale: State-Exact Trace-Preserving Deletion in Billion-Parameter Language Models* · Abdullah X · 2025 (v2 2026) · arXiv · 2508.12220 | "reconstructs a separately executed trace oracle bit-for-bit in model and optimizer state"; Pythia 2.8B matches all 2,775,208,960 elements; Llama 3.2 1B exact; Pythia 160M exact | Code on GitHub (v0.3.1). |
| C24 | Marin 32B: spikes traced to attention; QK-Norm removed them | **CORRECTED** | *Marin 32B Retro* · marin-community · marin.readthedocs.io/en/latest/reports/marin-32b-retro/ (no date on page) | "rooted in the model's attention stack rather than in optimizer state"; "the loss spikes disappeared entirely"; necromancy "relapsed into spikes" | See correction 5. Spikes occurred at 70k–80k steps of `exp1295_32b`. |
| C25 | Central Flows eq. 34 RMSProp ν relaxation (ICLR 2025); Cohen 2207.14484 38/η | CONFIRMED | *Understanding Optimization in Deep Learning with Central Flows* · Jeremy M. Cohen · 2024 · ICLR 2025 · 2410.24206 — *Adaptive Gradient Methods at the Edge of Stability* · Jeremy M. Cohen · 2022 · arXiv (no venue on page) · 2207.14484 | §5.2 Eq. 34: dν/dt = ((1−β2)/β2)[∇L⊙² + (4ν/η²)⊙diag Σ(t) − ν]; 2207.14484 abstract: "For Adam with step size η and β1 = 0.9, this stability threshold is 38/η." | The ν coefficient is (1−β2)/β2, and the target includes the oscillation term. |

## BibTeX (CONFIRMED items only)

```bibtex
@article{zhong2024swift,
  author = {Zhong, Yuchen and Sheng, Guangming and Liu, Juncheng and Yuan, Jinhui and Wu, Chuan},
  title = {{SWIFT}: Expedited Failure Recovery for Large-scale {DNN} Training},
  journal = {IEEE Transactions on Parallel and Distributed Systems},
  volume = {35}, number = {9}, year = {2024},
  eprint = {2302.06173}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2302.06173}
}
@inproceedings{biswas2026protected,
  author = {Biswas, Shrey},
  title = {Recent Training is Protected by {Adam}'s Second Moment},
  booktitle = {OPT 2026: Optimization for Machine Learning (NeurIPS 2026 Workshop)},
  year = {2026},
  url = {https://openreview.net/forum?id=VJFfO4tLMB}
}
@misc{eleuther2023neoxckpt160m,
  author = {{EleutherAI}},
  title = {neox-ckpt-pythia-160m-deduped-v1: final-step {GPT-NeoX} checkpoint with {ZeRO} optimizer states},
  year = {2023},
  note = {Year = Pythia suite release; repo upload date not verified},
  url = {https://huggingface.co/EleutherAI/neox-ckpt-pythia-160m-deduped-v1}
}
@inproceedings{vanderwal2025polypythias,
  author = {van der Wal, Oskar and Lesci, Pietro and Muller-Eberstein, Max and Saphra, Naomi and Schoelkopf, Hailey and Zuidema, Willem and Biderman, Stella},
  title = {{PolyPythias}: Stability and Outliers across Fifty Language Model Pre-Training Runs},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2025},
  eprint = {2503.09543}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2503.09543}
}
@misc{xu2026stored,
  author = {Xu, Qinyang},
  title = {Stored in Optimizer State, Valued by Later Training: A Causal Account of Subliminal Trait Transfer},
  year = {2026},
  eprint = {2608.20442}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2608.20442}
}
@misc{xie2026qkchannel,
  author = {Xie, Shuxiao and Xie, Shuyang and Cao, Yuan and Ran, Dezhi and Yang, Wei and Xie, Tao},
  title = {One {QK} Channel, Many Sources: Tracing Low-Precision Attention Collapse},
  year = {2026},
  eprint = {2608.02091}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2608.02091}
}
@misc{tang2026phasediagram,
  author = {Tang, Gaoxiang and Chen, Huanran and Liu, Ziming},
  title = {Beyond Quadratic Loss: The Stability Phase Diagram of {Adam}},
  year = {2026},
  eprint = {2609.18314}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2609.18314}
}
@inproceedings{altenbernd2026sdc,
  author = {Altenbernd, Anton and Wiesner, Philipp and Kao, Odej},
  title = {Exploring Silent Data Corruption as a Reliability Challenge in {LLM} Training},
  booktitle = {IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGrid)},
  year = {2026},
  eprint = {2604.00726}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2604.00726}
}
@inproceedings{tyagi2026llmprism,
  author = {Tyagi, Abhishek and Hukerikar, Saurabh and Saxena, Nirmal and Huang, Yanxiang and Shirvani, Philip and Tung, Chung-Hsuan and Zhu, Yuhao},
  title = {{LLM-PRISM}: Characterizing Silent Data Corruption from Permanent {GPU} Faults in {LLM} Training},
  booktitle = {International Conference for High Performance Computing, Networking, Storage, and Analysis (SC)},
  year = {2026},
  eprint = {2604.10390}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2604.10390}
}
@misc{donaghy2026open1b,
  author = {Donaghy, John and Wilcox, Brian and Ersoy, O{\u{g}}uzhan and Rastogi, Shikhar and St Arnaud, Adam and Titov, Alexey and Greenberg, Jordan and Fielding, Ben and Grieve, Harry},
  title = {{OPEN-1B}: A Fully Auditable Training Run},
  year = {2026},
  eprint = {2609.17380}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2609.17380}
}
@misc{aithal2026megatrondet,
  author = {Aithal, Ashwath and Harper, Eric and Bhavani, Santosh and Xu, Yan and Li, Zhiyu},
  title = {Scale Bitwise-Deterministic Pretraining with {NVIDIA} {Megatron} {Core}},
  year = {2026},
  howpublished = {NVIDIA Technical Blog, Oct 6, 2026},
  url = {https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/}
}
@misc{wortsman2023stable,
  author = {Wortsman, Mitchell and Dettmers, Tim and Zettlemoyer, Luke and Morcos, Ari and Farhadi, Ali and Schmidt, Ludwig},
  title = {Stable and low-precision training for large-scale vision-language models},
  year = {2023},
  eprint = {2304.13013}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2304.13013}
}
@misc{kassinos2025kourkoutas,
  author = {Kassinos, Stavros C.},
  title = {{Kourkoutas-Beta}: A Sunspike-Driven {Adam} Optimizer with Desert Flair},
  year = {2025},
  eprint = {2508.12996}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2508.12996}
}
@misc{zhu2026afor,
  author = {Zhu, Hongyu and Chen, Lin and Chen, Jing and Zhou, Yuting and Shang, Mingsheng},
  title = {Adaptive Forgetting for Nonstationary Optimization: Towards Robust {EEG} Decoding},
  year = {2026},
  eprint = {2609.24233}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2609.24233}
}
@misc{guo2026delayed,
  author = {Guo, Jinhui},
  title = {Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions},
  year = {2026},
  eprint = {2608.24593}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2608.24593}
}
@misc{kim2026secondmoment,
  author = {Kim, Jeonseong},
  title = {Second-Moment Memory in Coordinatewise {Adam}},
  year = {2026},
  eprint = {2608.15824}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2608.15824}
}
@inproceedings{zhou2026hiddenratio,
  author = {Zhou, Yihe and Zhu, Tongtian and Huo, Yingxiao and Dash, Satya P. and Wang, Can and Kaski, Samuel and Sun, Mingfei},
  title = {The Hidden Ratio in {Adam}: Stable Structure, Compression, and Sign Dynamics},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2026},
  url = {https://openreview.net/forum?id=GZha6judfp}
}
@misc{topollai2026quantstale,
  author = {Topollai, Kristi and Choromanska, Anna},
  title = {Understanding Quantization of Optimizer States in {LLM} Pre-training: Dynamics of State Staleness and Effectiveness of State Resets},
  year = {2026},
  eprint = {2603.16731}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2603.16731}
}
@misc{gemini2023,
  author = {{Gemini Team, Google}},
  title = {Gemini: A Family of Highly Capable Multimodal Models},
  year = {2023},
  eprint = {2312.11805}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2312.11805}
}
@misc{jiang2024crab,
  author = {Jiang, Yu and Shen, Jiyuan and Liu, Ziyao and Tan, Chee Wei and Lam, Kwok-Yan},
  title = {Towards Efficient and Certified Recovery from Poisoning Attacks in Federated Learning},
  year = {2024},
  eprint = {2401.08216}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2401.08216}
}
@misc{marek2025smallbatch,
  author = {Marek, Martin and Lotfi, Sanae and Somasundaram, Aditya and Wilson, Andrew Gordon and Goldblum, Micah},
  title = {Small Batch Size Training for Language Models: When Vanilla {SGD} Works, and Why Gradient Accumulation Is Wasteful},
  year = {2025},
  eprint = {2507.07101}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2507.07101}
}
@misc{kumar2025zclip,
  author = {Kumar, Abhay and Owen, Louis and Chowdhury, Nilabhra Roy and G{\"u}ra, Fabian},
  title = {{ZClip}: Adaptive Spike Mitigation for {LLM} Pre-Training},
  year = {2025},
  eprint = {2504.02507}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2504.02507}
}
@inproceedings{wang2026adagc,
  author = {Wang, Guoxia and Li, Shuai and Chen, Congliang and Zeng, Jinle and Yang, Jiabin and Yu, Dianhai and Ma, Yanjun and Shen, Li},
  title = {{AdaGC}: Enhancing {LLM} Pretraining Stability via Adaptive Gradient Clipping},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2026},
  eprint = {2502.11034}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2502.11034}
}
@inproceedings{huang2026gradientstabilizer,
  author = {Huang, Tianjin and Wang, Zhangyang and Hu, Haotian and Zhang, Zhenyu and Jin, Gaojie and Li, Xiang and Shen, Li and Shang, Jiaxing and Chen, Tianlong and Li, Ke and Liu, Lu and Wen, Qingsong and Liu, Shiwei},
  title = {{GradientStabilizer}: Fix the Norm, Not the Gradient},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2026},
  eprint = {2502.17055}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2502.17055}
}
@inproceedings{huang2025spam,
  author = {Huang, Tianjin and Zhu, Ziquan and Jin, Gaojie and Liu, Lu and Wang, Zhangyang and Liu, Shiwei},
  title = {{SPAM}: Spike-Aware {Adam} with Momentum Reset for Stable {LLM} Training},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2025},
  eprint = {2501.06842}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2501.06842}
}
@misc{abdullahx2025unlearning,
  author = {X, Abdullah},
  title = {Unlearning at Scale: State-Exact Trace-Preserving Deletion in Billion-Parameter Language Models},
  year = {2025},
  eprint = {2508.12220}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2508.12220}
}
@inproceedings{cohen2025centralflows,
  author = {Cohen, Jeremy M. and Damian, Alex and Talwalkar, Ameet and Kolter, J. Zico and Lee, Jason D.},
  title = {Understanding Optimization in Deep Learning with Central Flows},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2025},
  eprint = {2410.24206}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2410.24206}
}
@misc{cohen2022adaptiveeos,
  author = {Cohen, Jeremy M. and Ghorbani, Behrooz and Krishnan, Shankar and Agarwal, Naman and Medapati, Sourabh and Badura, Michal and Suo, Daniel and Cardoze, David and Nado, Zachary and Dahl, George E. and Gilmer, Justin},
  title = {Adaptive Gradient Methods at the Edge of Stability},
  year = {2022},
  eprint = {2207.14484}, archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2207.14484}
}
```

BibTeX is omitted for the CORRECTED items: C5 Liu & Li 2608.19762, C13 Bai et al. 2506.04805 (PMLR 306:5566–5601), C17 ABD 2007, C21 Ma et al. ACL 2025, and C24 Marin retro. The sources themselves are real and their metadata is in the table. Add their entries once the plan text is fixed.

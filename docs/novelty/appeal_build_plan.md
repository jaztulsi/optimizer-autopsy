# Appeal prototype: build plan

*Written 2026-11-12. A plan only: no code is written and nothing has run. It turns `rml3_method.md` §1 into a unit
that can be built, after reading the harness (`harness/trunk.py`, `harness/fork.py`, `autopsy/undo.py`,
`spikes/induce.py`, `model/nanogpt.py`). Where the two files differed, §1 was updated to match this one.
**[I]** marks my inference.*

## 1. Scope

**In this unit**
1. `research/autopsy/appeal.py`: Appeal as two training-loop hooks. No change to the loop.
2. A `hard_rows` spike recipe in `research/spikes/induce.py`: the control that should be restored.
3. `research/tests/test_appeal.py`: CPU tests that run in CI.

**Not in this unit**
- Wiring the arms into the GPU driver (`autopsy/mve.py`), and any GPU run.
- A rank-calibrated (conformal) flag threshold, and ghost gradient norms as the score.
- Per-row losses exposed by the loop (that changes `train_forward` and needs the `Δ==0` gate re-run).
- Gradient accumulation above 1.
- The tournament (K2) and the ceiling (K3).

## 2. Where it lives

`research/autopsy/`, next to the RML undo it reuses. `research/repair/` is listed under Eddie's proposed role in
`PLAN_V8.md` §11.1 ("the C3 operators beyond RML"), so the prototype stays out of it until Eddie says where he
wants it. Moving one file later is cheap.

## 3. What reading the code settled

| # | Decision | The code fact behind it | Change from the first draft of §1 |
|---|---|---|---|
| 1 | Two closures, a `pre_step` and a `pre_opt_step`, share one state dict. The state resets when the step counter goes backwards | `fork.run_branch` deep-copies `pre_opt_step`. A function copies to itself, so closures keep sharing state; a bound method would not (`undo.py` says the same) | none |
| 2 | The spike recipe runs first and Appeal second, through `fork.compose` | `compose` is order-sensitive. Appeal reads `ctx["batch"]` if a recipe set it, and otherwise fetches the aligned slot with `get_batch`. The data cursor never moves | none |
| 3 | Rows are flagged by a robust z-score inside their own batch (median and median absolute deviation of that batch's row losses) | A fork starts at the spike step (`mve` forks from the snapshot at `inject_step`), so there is no history to score against | **changed**: was a running window with a conformal threshold |
| 4 | If more than half the rows are flagged, Appeal does nothing | The median is no longer robust, and masking every row makes the loss a mean over zero tokens. Whole-batch events belong to skip-step and the tournament | new |
| 5 | Withholding sets the flagged rows' targets to −1 | The loss ignores target −1 and averages over the rest (`nanogpt.py`), so Adam is fed the mean of the kept rows: a trimmed mean, not a partial sum | restore formula now weights by row counts |
| 6 | The flagged rows' gradient comes from a separate forward and backward pass on those rows, taken with `torch.autograd.grad` | `pre_step` fires before `zero_grad`; this route never touches `p.grad` | none |
| 7 | The reference is taken at the flagged step: eight random groups of kept rows, each the size of the flagged group | The verdict needs a reference K steps later, and the fork has no earlier steps to build one from. Same step, same weights, same size and same later batches make it a matched control | **changed**: was a periodic shadow record |
| 8 | Evidence is a cosine in Adam's scaling, summed over the K later steps | Hard rows have large gradients. A score divided by the record's squared norm would penalise them for size alone | **changed**: was divided by the squared norm |
| 9 | φ = clip((p − 0.05) / 0.20, 0, 1), where p is the share of shadow groups scoring lower | With eight groups, φ is 0, 0.375 or 1 | fixed on 2026-11-12 (the draft could go negative) |
| 10 | The restore reuses `undo.undo` with the entry (ĝ_0 − ĝ_φ, ĝ_0² − ĝ_φ²) | That function already subtracts (1 − β)·β^age times the entry and keeps v at or above zero. Fired at step s + K before the update, the moments are those left by step s + K − 1, so the decay power is K − 1 | **changed**: the draft said β^K |
| 11 | The kept mean before the clip is ĝ_0 / c, with c = min(1, clip / (norm + 1e-6)) from `ctx["grad_norm"]`. ĝ_φ is clipped again with the same formula | This is what `clip_grad_norm_` applies | detail |
| 12 | One parameter, `force_phi`, gives two control arms: 0 is trimming without appeal; 1 is "delayed keep" (always restore) | Delayed keep prices the K-step delay by itself | new arm |
| 13 | A `flag_fn` override lets the caller name the flagged rows | It separates testing the verdict from testing the detector. The first experiment knows which rows it injected | new |
| 14 | Gradient accumulation above 1 raises an error | The batch override applies to every micro-batch | limit |
| 15 | `train_forward`, `determinism.py`, `preflight.py` and both requirements files are not touched | hooks only | none |

## 4. Interface (names only)

| Name | Does |
|---|---|
| `row_losses(model, x, y)` | Mean loss per row. One forward pass, no gradients, eval mode, so it draws nothing from the RNG |
| `flag_rows(losses, z)` | Boolean mask of rows whose robust z-score exceeds `z` |
| `make(data_dir, lag=10, z=6.0, n_shadow=8, lo=0.05, hi=0.25, seed=0, flag_fn=None, force_phi=None)` | Returns `(pre_step, pre_opt_step, state)` |
| `state["verdicts"]` | One entry per closed record: step, rows, n, φ, p, the record's score, the shadow scores |
| `state["batch_events"]` | Steps where more than half the batch was flagged and Appeal stood aside |

A branch in the battery would then be `compose(recipe.pre_step, pre_step)` with `pre_opt_step` beside it. That
wiring is a later unit.

## 5. The `hard_rows` recipe

- Replaces the first ceil(frac · B) rows of the step's real batch with the highest-loss rows of a pool of real
  training rows, scored at the current weights. Default: 25% of rows, from a pool of four batches.
- The pool comes from data slots beyond `max_steps`, which the run never trains on otherwise. It never comes from
  the validation split, because the held-out probe has to stay held out.
- No random numbers are drawn; the choice is fixed by the weights.
- The clean counterfactual is unchanged: the same run with the recipe left out.
- The recipe list in `induce._selfcheck` grows from 9 to 10.

## 6. Tests (CPU, in CI) and what each would catch

| # | Test | Would catch |
|---|---|---|
| 1 | The flag rule on hand-made losses: one outlier is flagged; with more than half outliers the hook leaves the batch alone | threshold and robustness mistakes |
| 2 | A run with the hooks on and nothing flagged is bitwise identical to a run without them | the scoring pass disturbing training (mode, RNG, stray gradients) |
| 3 | A masked step leaves the same weights as a step fed only the kept rows | wrong masking or normalisation |
| 4 | The restore algebra, for `force_phi` 0.5 and 1 and for Appeal's own verdict, with the clip off and with it active: at step s + K, m and v change by exactly (1 − β)·β^(K−1) times (ĝ_φ − ĝ_0) and (ĝ_φ² − ĝ_0²), where ĝ_φ is rebuilt in the test from separate full-batch and kept-row backward passes and torch's own clip | wrong row weights, wrong decay power, wrong sign, wrong clip handling |
| 5 | `force_phi = 0` leaves m and v bitwise untouched at step s + K | a trim arm that edits state |
| 6 | The cosine helper and the φ mapping on hand-made numbers | arithmetic slips |
| 7 | Reusing the hooks for a second run gives the same verdict list as the first | state leaking between branches |
| 8 | Gradient accumulation of 2 raises | silent misuse |
| 9 | `hard_rows`: the replaced rows are the top rows of the pool by a loss computed separately in the test; other rows are the real batch; outside the window nothing changes; a missing data directory raises | a recipe that injects the wrong rows |

**Not a CI question, on purpose:** whether junk rows are dropped and hard rows restored. That is statistical and
belongs to the GPU experiment (A1, A2 in `rml3_method.md` §1.5). A green CI run means the mechanics are right; it
does not mean Appeal works.

## 7. Order of work

1. The pure pieces (`row_losses`, `flag_rows`, cosine, verdict) with tests 1 and 6.
2. The two hooks with tests 2 to 5, 7 and 8.
3. `hard_rows` with test 9 and the self-check list.
4. The status row in `research/README.md`.
5. A pull request; done when `ruff check` and `pytest` are green in CI, with tests 2 and 4 passing.

After that, as separate units: the battery arms in `mve` with a dry run; a pre-registration section carrying A1 to
A5; the GPU run, with the session go-ahead.

## 8. Cost

- Every step: one extra forward pass without gradients (about a third of a step [I]).
- A flagged step: one small backward pass for the flagged rows and one per shadow group.
- Memory: about eleven gradient-sized vectors for K steps at the defaults (the flagged share, eight shadow groups,
  ĝ_0 and the scaling P). Fine at proxy scale; at 124M it would need sketches.

## 9. Open questions

1. Whether Eddie wants the file under `research/repair/`.
2. The lag K. At β1 = 0.9, a correction applied 10 steps late reaches m scaled by 0.9⁹ = 0.39 and v by
   0.95⁹ = 0.63. More lag is more evidence and a weaker, staler restore. Suggested sweep: K = 3, 10, 20.
3. Shadow groups are fed to Adam at the flagged step and the withheld share is not. That tilts the verdict
   slightly toward restoring [I].
4. A masked step logs the kept rows' mean loss, so a loss-based detector sees a smaller spike on that step.
5. Hard rows have to exceed the flag threshold for the appeal to be tested at all. If z = 6 does not flag them,
   the first experiment names the injected rows through `flag_fn`.
6. A batch that is garbage in every row (`corrupt_batch`) passes through Appeal untouched. That is by design
   (decision 4), and it is the case the tournament is for.

## 10. The forecast layer (added 2026-11-12 at Jasraj's request; design only)

**Prior art for this section was not checked.** Five searches were attempted and all failed at the usage limit.
Nothing here may be called novel until they are run. Forecasting spikes from AdamW state is already published
(Kim S. 2026, *Transport Instability Coordinates*; `method.md` §5), so forecasting itself is not claimed.

Three small predictors, each with one job. All are cheap: free statistics and a tiny model.

| | Predicts | Inputs | What it controls |
|---|---|---|---|
| **F1 risk** (when) | the chance of a spike in the next few steps | the free statistics in `method.md` §2.6: RMS(g²/v), the share of coordinates with \|m/√v\| > 1, cos(g, m), gradient-norm and loss z-scores, lag-1 gradient autocorrelation, the learning rate. Logistic regression | (a) keeps a snapshot for the tournament; (b) lowers Appeal's flag threshold; (c) starts a tournament **before** the spike over {do nothing, learning-rate cooldown}. "Do nothing" is on the menu, so a false alarm costs compute and no loss |
| **F2 row prior** (which rows) | how odd a row is as text, at the flagged step itself | a tiny independent model of the data, such as bigram counts from the training shard | shifts Appeal's cut-offs: junk n-grams are extreme under it, hard real text is not. It also routes: a high loss under the live model alone points to the state, which is the tournament's case |
| **F3 remedy policy** (which remedy) | which menu item will win | the same statistics as F1, labelled by the tournament's own outcomes (exact counterfactual labels from forks) | orders the menu, and skips the tournament only when confident |

- **Labels for F1:** induced spikes that build up (`lr_bump` with width above 1, `tiny_eps`, multi-step
  `repeat_ngram`), scored with the lead-time code already in `spikes/tune_detector.py`. The baseline is the existing
  `OnlineDetector`.
- **Wrong if (proposals):** F1 fires no earlier than the existing detector at an equal false-alarm rate, tested with
  one recipe held out; F2 does not raise A1 and A2 above the later-batch evidence alone; F3's first pick has a
  higher regret than always choosing "do nothing".
- **Limits [I]:** a pure bad batch has no warning sign, so F1 cannot see it and the flag at the step itself does
  that job. The proxy has few natural spikes to learn from. Filtering data by a small reference model's perplexity
  exists as a pruning method (from memory, unverified).
- **File, as a later unit:** `research/autopsy/forecast.py` (features in `on_step`, a pure scoring function).

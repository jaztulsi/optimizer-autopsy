# TODO — things I need YOU to do

This file is where I put anything I can't do from inside Claude Code: create an account, get an API
key/token, run something on a GPU, confirm a value, etc. Each item says **what**, **why**, **how**,
and **where**.

**The loop:** do an item → type `done` (or `done <n>`) in the chat → I verify → I either remove it
or tell you it's wrong and to redo it. This file is **never empty**: if I have nothing for you, the
list below will just say *"Nothing left for you to do!"*

---

## Open items

1. **Task 5 DoD — run the trunk on a Kaggle GPU** (not your Mac).
   - **What:** in a Kaggle notebook with the repo cloned, run:
     `python -m research.harness.trunk --config research/experiments/proxy/config.yaml --steps 200`
   - **Why:** confirms the training loop actually trains on the real proxy shard (loss goes down)
     — the last thing left to close out Task 5. We keep compute off your Mac.
   - **How:** paste that command in a Kaggle cell (GPU T4 on). It prints
     `trunk done: 200 steps, loss X -> Y`.
   - **Where:** paste the `loss X -> Y` line back here and type `done 1`.

---

## Done ✅

- **Perfect-replay GPU check** — `bitwise replay OK on cuda: 50 steps, max|Δ|=0`. Task 4 DoD met.
- **HF artifacts repo** — `jaztulsi/optimizer-autopsy-artifacts` created (private, Dataset).
- **Accounts** — Kaggle, Hugging Face, W&B, Google/Colab created.
- **Secrets** — `HF_TOKEN` + `WANDB_API_KEY` stored in Kaggle Secrets **and** Colab Secrets
  (Notebook access ON in both).
- **Kaggle GPU + Internet** — phone-verified; `GPU T4 ×2` + Internet confirmed working (repo clones).
- **Version pins** — `check_env()` prints `GPU: Tesla T4` / `check_env OK` on Kaggle;
  `requirements.txt` now matches the real image (torch 2.10.0, numpy 2.0.2, …). *Task 1 DoD met.*

---

*When you finish any item, type `done <number>` in the chat.*

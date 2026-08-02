# TODO — things I need YOU to do

This file is where I put anything I can't do from inside Claude Code: create an account, get an API
key/token, run something on a GPU, confirm a value, etc. Each item says **what**, **why**, **how**,
and **where**.

**The loop:** do an item → type `done` (or `done <n>`) in the chat → I verify → I either remove it
or tell you it's wrong and to redo it. This file is **never empty**: if I have nothing for you, the
list below will just say *"Nothing left for you to do!"*

---

## Open items

### 1. Run the "perfect replay" test on a Kaggle GPU
- **Why:** it already passes on your laptop (CPU), but GPUs do math slightly differently. The plan
  requires it to pass on a real GPU too before we trust anything built on top of it.
- **How/where:** open a Kaggle notebook with **GPU T4 + Internet on**, first cell, run:
  ```python
  import os; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
  %cd /kaggle/working
  !rm -rf optimizer-autopsy
  !git clone https://github.com/jaztulsi/optimizer-autopsy.git
  %cd optimizer-autopsy
  !python -m research.tests.test_determinism
  ```
- **Done when:** you paste me the output. Good result looks like:
  `bitwise replay OK on cuda: 50 steps, max|Δ|=0 at every step`.

---

## Done ✅

- **HF artifacts repo** — `jaztulsi/optimizer-autopsy-artifacts` created (private, Dataset).
- **Accounts** — Kaggle, Hugging Face, W&B, Google/Colab created.
- **Secrets** — `HF_TOKEN` + `WANDB_API_KEY` stored in Kaggle Secrets **and** Colab Secrets
  (Notebook access ON in both).
- **Kaggle GPU + Internet** — phone-verified; `GPU T4 ×2` + Internet confirmed working (repo clones).
- **Version pins** — `check_env()` prints `GPU: Tesla T4` / `check_env OK` on Kaggle;
  `requirements.txt` now matches the real image (torch 2.10.0, numpy 2.0.2, …). *Task 1 DoD met.*

---

*When you finish any item, type `done <number>` in the chat.*

# TODO — things I need YOU to do

This file is where I put anything I can't do from inside Claude Code: create an account, get an API
key/token, run something on a GPU, confirm a value, etc. Each item says **what**, **why**, **how**,
and **where**.

**The loop:** do an item → type `done` (or `done <n>`) in the chat → I verify → I either remove it
or tell you it's wrong and to redo it. This file is **never empty**: if I have nothing for you, the
list below will just say *"Nothing left for you to do!"*

---

## Open items

### 1. Create the private HF artifacts repo
- **Why:** snapshots, checkpoints, and tokenized shards all live here (not on your laptop).
- **How/where:** <https://huggingface.co/new-dataset> → name `optimizer-autopsy-artifacts` →
  **Private** → Create. (Must be a **Dataset** repo, not a Model repo.)
- **Done when:** `jaztulsi/optimizer-autopsy-artifacts` exists and is private.

### 2. Phone-verify Kaggle + enable Internet/GPU
- **Why:** Kaggle blocks Internet (`kernelSessions.enableInternet` denied) until your account is
  phone-verified — and without Internet you can't clone the repo or pull from HF.
- **How/where:** <https://www.kaggle.com/settings> → **Phone Verification**. Then in a notebook's
  right sidebar → **Session options** → turn **Internet ON** and **Accelerator → GPU (T4/P100)**.
- **Done when:** a notebook cell running `!pip --version` with Internet on doesn't error.

### 3. Finalize the version pins against Kaggle's real image  ← *unblocks Task 1's DoD*
- **Why:** `requirements.txt` currently holds my best-guess pins. `check_env()` asserts installed ==
  pinned, so the numbers must match Kaggle's actual GPU image or every real run refuses to start.
- **Needs:** items 1 and 2 done first (Internet on).
- **How/where:** open a Kaggle notebook **with GPU + Internet on**, first cell, run:
  ```python
  import os; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
  !git clone https://github.com/jaztulsi/optimizer-autopsy.git
  %cd optimizer-autopsy
  !python -m research.harness.preflight
  ```
  It will either print `check_env OK` (perfect — nothing to do) or list lines like
  `torch: installed 2.5.1 != pinned 2.4.0`.
- **Done when:** you paste that output into the chat. I'll update `requirements.txt` to match.

---

## Done ✅

- **Accounts** — Kaggle, Hugging Face, W&B, Google/Colab created.
- **Secrets** — `HF_TOKEN` + `WANDB_API_KEY` stored in Kaggle Secrets **and** Colab Secrets
  (Notebook access ON in both).

---

*When you finish any item, type `done <number>` in the chat.*

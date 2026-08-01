# TODO — things I need YOU to do

This file is where I put anything I can't do from inside Claude Code: create an account, get an API
key/token, run something on a GPU, confirm a value, etc. Each item says **what**, **why**, **how**,
and **where**.

**The loop:** do an item → type `done` (or `done <n>`) in the chat → I verify → I either remove it
or tell you it's wrong and to redo it. This file is **never empty**: if I have nothing for you, the
list below will just say *"Nothing left for you to do!"*

---

## Open items

### 1. Create the free accounts (one-time)
- **Why:** everything heavy runs in the cloud; the repo on your laptop is just text.
- **How/where:**
  - Kaggle → <https://kaggle.com> (then Settings → Phone Verify to unlock GPU).
  - Hugging Face → <https://huggingface.co/join>
  - Weights & Biases → <https://wandb.ai>
  - Google (for Colab) and Zotero (refs) — if you don't already have them.
- **Done when:** you can log into all of them.

### 2. Create tokens and store them as PLATFORM SECRETS (never in git)
- **Why:** the code loads `HF_TOKEN` / `WANDB_API_KEY` from the Kaggle/Colab secret store, not files.
- **How/where:**
  - HF **write** token: <https://huggingface.co/settings/tokens> → New token (role: Write).
  - W&B key: <https://wandb.ai/authorize>.
  - Kaggle → any notebook → **Add-ons → Secrets** → add `HF_TOKEN` and `WANDB_API_KEY`.
  - Colab → **🔑 (left sidebar)** → add the same two secrets.
- **Done when:** both secrets exist in Kaggle **and** Colab. (Do NOT paste them in the chat.)

### 3. Create the private HF artifacts repo
- **Why:** snapshots, checkpoints, and tokenized shards all live here (not on your laptop).
- **How/where:** <https://huggingface.co/new> → name `optimizer-autopsy-artifacts` → **Private** →
  set type to **Dataset**.
- **Done when:** `jaztulsi/optimizer-autopsy-artifacts` exists and is private.

### 4. Finalize the version pins against Kaggle's real image  ← *unblocks Task 1's DoD*
- **Why:** `requirements.txt` currently holds my best-guess pins. `check_env()` asserts installed ==
  pinned, so the numbers must match Kaggle's actual GPU image or every real run refuses to start.
- **How/where:** open a Kaggle notebook **with GPU on**, first cell, run:
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

*When you finish any item, type `done <number>` in the chat.*

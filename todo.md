# TODO — things I need YOU to do

This file is where I put anything I can't do from inside Claude Code: create an account, get an API
key/token, run something on a GPU, confirm a value, etc. Each item says **what**, **why**, **how**,
and **where**.

**The loop:** do an item → type `done` (or `done <n>`) in the chat → I verify → I either remove it
or tell you it's wrong and to redo it. This file is **never empty**: if I have nothing for you, the
list below will just say *"Nothing left for you to do!"*

---

## Open items

**Nothing left for you to do!** 🎉

(I can now drive Kaggle myself, headless — the Kaggle CLI is wired up on your Mac, so I push code
and run it on a free GPU and pull results back without you clicking anything. Compute never touches
your machine.)

---

## Done ✅

- **Task 5 DoD — trunk trains on a Kaggle GPU** — I ran it headless on a Tesla T4:
  `trunk done: 200 steps, loss 10.851 -> 4.730`. Loss more than halved → the training loop learns.
  Task 5 met.
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

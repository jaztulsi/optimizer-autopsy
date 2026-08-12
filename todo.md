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

## 🚀 Tools to make this faster/easier (all FREE — optional, grab when you want)

Legend: ✅ already wired · ⚡ high value · 🔹 nice-to-have

### CLIs / APIs (things that run on your Mac to drive the cloud)
- ✅ **Kaggle CLI** — I push code + run it on a free GPU + pull results, fully headless. Wired up.
- ✅ **GitHub CLI (`gh`)** — commit/push/PRs as jaztulsi. Already authed.
- ⚡ **Hugging Face login** (`hf auth login`) — lets runs **save/load checkpoints** to your
  `optimizer-autopsy-artifacts` box, so results survive after a Kaggle session ends. *What I need
  from you:* run `hf auth login` and paste your `HF_TOKEN` (or drop the token here).
- ⚡ **Weights & Biases** (`wandb`) — live **loss-curve charts** I can read remotely instead of
  digging through logs. *What I need:* say the word and I'll install it; you paste `WANDB_API_KEY`.

### Claude Code skills / plugins (make ME faster at this project)
- ⚡ **kaggle-skill** (github.com/shepsci/kaggle-skill) — a cleaner wrapper over the Kaggle CLI:
  dataset/model download, notebook execution, competition reports.
- ⚡ **Context7** — live, version-correct docs. Useful because our stack is version-pinned
  (torch 2.10, numpy 2.0.2) — stops stale-API mistakes.
- 🔹 **Semgrep plugin** — free security/bug scan on each change.
- 🔹 Browse more at github.com/composio-community/awesome-claude-plugins.

### MCP servers (already installed on your machine)
- ✅ **ruflo-core**, **ponytail**, **claude-in-chrome**, **Google Drive** — already available. Drive
  can be an alt artifact store if you'd rather not use Hugging Face.

**My recommendation:** the two with real payoff are **Hugging Face login** + **W&B** — together they
make every run's results and charts persist and be readable remotely. Everything else is polish.
Tell me when you want them and I'll set them up.

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

## 🧰 Make this the best it can be — free tools/skills/plugins to wire in

Grouped by payoff. Everything here is **free**. ✅ wired · ⚡ high value · 🔹 polish.

### Compute & artifacts (persist results, run bigger)
- ⚡ **Hugging Face login** — checkpoints/snapshots survive after a Kaggle session dies. *You:* `hf auth login`.
- ⚡ **Weights & Biases** — live loss curves I can read remotely. *You:* paste `WANDB_API_KEY`.
- 🔹 **Kaggle Datasets as cache** — pre-tokenized shards + spike sets as a private dataset, so runs
  skip re-prep and start in seconds.
- 🔹 **Colab as a 2nd free GPU** — parallel proxy iteration while Kaggle runs the big job.

### CI / quality gates (catch breakage before a GPU run wastes an hour)
- ⚡ **GitHub Actions** (free for public repos) — on every push: `pytest`, import check, and the
  `test_no_secrets_in_git` guard. Red X before you ever push to Kaggle.
- ⚡ **ruff** — one fast linter+formatter for all of `research/`. Add `ruff.toml`, run in CI.
- 🔹 **pre-commit** — run ruff + secret-scan locally before each commit.
- 🔹 **Codecov (free OSS tier)** — coverage badge on the harness/localizer tests.

### Security / secret hygiene (you asked for this)
- ⚡ **gitleaks** or **GitHub secret scanning** (free, public repos) — blocks a token from ever landing.
- 🔹 **Semgrep** (free tier / plugin) — cheap bug + security scan each change.
- 🔹 **Dependabot** — free auto-PRs when a pinned dep has a fix.

### Claude Code skills / plugins (make ME faster here)
- ⚡ **kaggle-skill** (github.com/shepsci/kaggle-skill) — cleaner wrapper over the Kaggle CLI.
- ⚡ **Context7 MCP** — version-correct docs for our pinned stack (torch 2.10 / numpy 2.0.2).
- 🔹 **graphify** (`/graphify`) — turn the research code + BUILD_PLAN into a queryable knowledge graph.
- 🔹 **/code-review** & **/security-review** — run before each push.
- 🔹 **/schedule** — a nightly cloud agent that re-runs the smoke test + posts status.

### Repo polish (makes it look "the best")
- ⚡ **README badges live** — CI status, license, last-run loss (already scaffolded in `README.md`).
- 🔹 **LICENSE file** (MIT/Apache-2.0) — pick one, I'll add it.
- 🔹 **CITATION.cff** — so the eventual paper/repo is citable.
- 🔹 **GitHub repo topics + description** — `pytorch`, `optimizer`, `interpretability`, `training-dynamics`.
- 🔹 **Issue/PR templates** — if you ever open it to collaborators.

**My pick for biggest payoff, in order:** HF login + W&B (results persist & are readable) →
GitHub Actions CI (never push broken code to a GPU) → gitleaks (secret safety) → a LICENSE file.
Say the word on any and I'll set it up.

---

*When you finish any item, type `done <number>` in the chat.*

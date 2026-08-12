# Operating rules for this project

These are standing rules for any AI assistant working in this repo. Follow them every task.

1. **Verify your own work.** After any change or remote run, read the actual log/output and
   confirm it did what you claim. Never assume success. Report failures plainly with the output.

2. **Remote-first & efficient.** Keep the workflow headless. Proactively tell the user which
   API/CLI to connect to make things more automated (Kaggle, Hugging Face, W&B, `gh`, …), and
   always choose the most efficient/effective path.

3. **Free only.** Every service, tool, or compute recommended must be free (free tier / free GPU).
   Never recommend anything paid.

4. **NEVER run anything on the user's local machine — CPU, GPU, or otherwise.** No training,
   no eval, no model code, no data prep, no scripts. ALL compute runs on free remote GPU (Kaggle
   primary; Colab has no headless API, so prefer Kaggle). The only things allowed to run locally
   are the unavoidable plumbing to drive remote work: `git`, the `kaggle`/`gh` CLIs, and reading
   files. When in doubt, push it to Kaggle.

5. **Commit & push as jaztulsi, never Claude.** git author is already
   `jaztulsi <jaztulsi99@gmail.com>`. Do NOT add `Co-Authored-By: Claude` trailers or attribute
   anything to Claude.

## Remote GPU workflow (Kaggle)

Kaggle CLI is wired on the user's Mac at `~/Library/Python/3.13/bin/kaggle`.
Loop: edit code → **commit+push to `main`** (Kaggle clones the default branch from GitHub) →
`kaggle kernels push` a script kernel with `enable_gpu:true, enable_internet:true` → poll with
`kaggle kernels status` → pull results with `kaggle kernels output`.

Data shards (`*.bin`, `/data/`) are gitignored; the `research/data/` **package** is tracked.

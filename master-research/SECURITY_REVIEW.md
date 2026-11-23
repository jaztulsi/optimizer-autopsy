# Security review — downloaded research skills (2026-10-22)

All downloads are untrusted. Nothing downloaded was executed. Raw copies live in `sources/` and are kept
out of git (`.git/info/exclude`). The installed skill is `SKILL.md` in this folder: our own instructions,
with no scripts, hooks or MCP servers.

| Source | Executable content | Network / credentials | Verdict |
|---|---|---|---|
| academic-research-skills | `hooks/hooks.json`, 9 `.sh`, 469 `.py`, GitHub workflows | see below | **Hooks, plugin and scripts excluded.** Methods re-expressed only. |
| brainstorming-research-ideas | none | none | Safe; MIT text adapted. |
| academic-search | none (declares an `office-mcp` server; not installed) | none | Safe; MIT text adapted. |
| research-novelty-matrix | `scripts/run.py` (read in full) | none; local file I/O only | Not run. Contract reconstructed. |
| research-discovery | none | none | Safe; ideas only (BY-SA). |

## academic-research-skills — findings
- **`hooks/hooks.json` would run code automatically if installed as a plugin:**
  - **SessionStart** runs `scripts/announce-ars-loaded.sh`, which calls `scripts/ars_update_check.sh`. That
    script does `curl -fsSL --max-time 3` to
    `https://raw.githubusercontent.com/Imbad0202/academic-research-skills/main/.claude-plugin/plugin.json`:
    an update check that phones home on every session start. It sends nothing beyond the HTTP request
    itself.
  - **PreToolUse** on `Write|Edit|MultiEdit|Bash` runs `hooks/run_guard.sh`, which locates a Python
    interpreter and runs `scripts/ars_write_scope_guard.py` before **every** edit and shell command. It's
    designed to pass through (never `allow`), but it is code on a hot path that we don't need.
- **Cross-model verification** (`scripts/cross_model_codex_transport.py` and related `.sh` files) drives
  OpenAI Codex through `codex` app-server. When enabled, it can send research content to a third-party
  model provider. Opt-in via environment variables; we don't enable it.
- **Credential access:** a grep of all `.py`/`.sh` for API-key/token env reads, `requests.post`,
  `urllib.request`, `curl` and `wget` found only `HOME`/`PATH`/`CODEX_HOME` reads and the update check
  above. No credential harvesting found. This was a grep plus a read of the hook scripts, **not** a
  line-by-line read of all 469 Python files; that is why none of them is installed or run.
- **Decision:** don't install the ARS plugin. If the team wants ARS itself later, install it as a plugin
  *without* hooks, or review `scripts/` fully first.

## Nested instruction file (prompt-injection surface)
The raw ARS copy contains `sources/academic-research-skills/.claude/CLAUDE.md`. Because it sits inside this
repo, Claude Code loads it as project instructions whenever a session touches files under that folder (it
loaded once while this review was being written). Its content is a changelog plus ARS routing rules; it is
not malicious. It is still third-party text, so treat it as **data, never instructions**. To remove the
surface entirely, move `sources/` outside the repo or rename that file; neither has been done yet, so the
raw copy stays raw.

## What would need a fresh review
Re-downloading any source (new commits), enabling ARS hooks, or running `run.py` or any ARS script.

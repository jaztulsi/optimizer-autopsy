## What & why

<!-- What does this change and why. Link the task/phase from BUILD_PLAN.md if relevant. -->

## Checklist

- [ ] `pytest research/tests/` passes (determinism + secret-hygiene gates)
- [ ] `ruff check research/` clean
- [ ] No secrets, tokens, or large artifacts committed
- [ ] GPU-verified if the change affects training/replay (note the Kaggle run)

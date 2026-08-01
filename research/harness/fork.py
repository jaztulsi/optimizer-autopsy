"""Fork runs: branch off a trunk snapshot to test interventions counterfactually.

A fork loads (w, m, v) at step T, applies an intervention (repair operator or baseline),
and trains forward. Comparing forks isolates the causal effect of the intervention.
"""

# TODO: run_fork(snapshot, intervention, cfg) -> load snapshot, apply intervention, train N steps.
# TODO: short_fork / full_fork -> convenience wrappers for the 500-2000 step vs full-convergence runs.
# TODO: fork_matrix(snapshot, interventions, seeds) -> run the N x methods x seeds sweep.

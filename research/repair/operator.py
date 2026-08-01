"""Repair operator: the intervention applied to a snapshot before a fork.

Acts only on the localized (poisoned) groups: reset/rescale their optimizer moments, optionally
adjust weights, leaving the rest of the state untouched. This is what forks test causally.
"""

# TODO: repair(snapshot, localization, mode) -> new snapshot with edited (w, m, v) on target groups.
# TODO: modes: "reset_v", "rescale_v", "reset_m", ... -> the menu of surgical edits to compare.
# TODO: as_intervention(localization, mode) -> callable usable by harness.fork.run_fork.

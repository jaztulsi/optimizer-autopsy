"""K2 spike set: the curated manifest of confirmed spikes (induced + natural) the paper reports on.

Each entry pins everything the downstream consumers need, checked against the real interfaces (CLAUDE.md §7):
  * localizer (C2) scores a (w, m, v) snapshot at the detector trigger t0 against the pre-spike running
    average v̄ (`vbar_uri`) and matched NORMAL-step snapshots (PLAN V6 §3.6), with a fixed probe batch.
  * repair (C3) + forks (harness/fork.py) rebuild the cfg, data stream and spike hook to replay the spike
    from `fork_snapshot_uri`, with B* = the same run with the hook omitted.
  * attribution scores held-out val loss per branch and needs the evidence behind each label.

Snapshot step convention (harness/snapshot.py + fork.py): a snapshot's `step` is the NEXT step to run --
forks resume at `snap["step"]`. The detector trigger t0 is always >= inject_step, so a snapshot at t0
already contains the injected update(s). Replaying the spike in a fork therefore needs a SEPARATE
fork-point snapshot with step <= inject_step (natural spikes: <= t0). Both are stored.

Entries are built from capture runs (`spikes/capture.py` writes one run.json per run) by `build_k2`, which
scores each trigger with the run's pre-declared policy -- induced: `tune_detector.detection_qualifies`;
natural: the trigger must fall on a `natural.find_spikes` event. Pure Python + numpy (CPU-testable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

PROBE = {"split": "val", "idx": [0, 1, 2, 3], "batch": 16}  # fixed curvature/eval batches (PLAN V6 §3.5)
TUNING_SEED = 0  # seed 0 tunes the detector; a "confirmed" induced entry must come from a held-out seed
MATCH_BEFORE = 25  # a natural-run trigger counts for an event if start - MATCH_BEFORE <= t0 <= peak_step


@dataclass(frozen=True)
class SpikeEntry:
    """One confirmed (snapshot, label) spike."""

    # --- identity / cause ---
    entry_id: str  # e.g. "lr_bump-i160-s1-t161"
    source: str  # "induced" | "natural"
    recipe: str | None  # spikes.induce.RECIPES kind; None for natural spikes
    recipe_params: dict  # factor= / n= / targets= ... ; for natural spikes {"stress": preset}
    inject_step: int | None  # known cause step; None for natural spikes
    width: int | None
    seed: int

    # --- reproduce the run (cfg, data, B*) ---
    config_path: str  # e.g. "research/experiments/proxy/config.yaml"
    config_overrides: dict  # e.g. {"train.batch_size": 16, "seed": 1}
    data_preset: str  # data.prepare.PRESETS key, e.g. "proxy"
    data_revision: str  # the preset's pinned HF commit sha

    # --- detection ---
    t0: int  # detector trigger step
    peak_step: int
    peak_ratio: float  # peak / pre-spike baseline
    lead: int  # peak_step - t0
    detection_mode: str  # "predictive" (lead >= L) | "onset" (t0 <= peak; never called early warning)
    detector_params: dict  # operating point used: z, win, ema_mult, warmup
    trigger_stat: dict  # OnlineDetector.stat at t0: loss z-score, grad-norm / past EMA

    # --- the states consumers load ---
    t0_snapshot_uri: str  # (w, m, v, rng) after step t0 -- what the localizer scores
    fork_snapshot_uri: str  # what forks replay the spike from
    fork_step: int  # that snapshot's `step` (<= inject_step for induced, <= t0 for natural)
    vbar_uri: str  # pre-spike running average of v at the fork point
    normal_snapshot_uris: tuple[str, ...]  # matched non-spike steps for v̄ + thresholds (PLAN V6 §3.6)
    probe: dict = field(default_factory=lambda: dict(PROBE))

    # --- label + provenance ---
    label: str = "provisional"  # "confirmed" only after a held-out DoD pass
    evidence: str = ""  # run.json / results json backing the label
    hardware: dict = field(default_factory=dict)  # {"gpu", "torch", "backend": "cuda" | "rocm", "deterministic"}


@dataclass
class K2Manifest:
    version: int = 1
    entries: list[SpikeEntry] = field(default_factory=list)


def validate(e: SpikeEntry, *, L: int = 2) -> None:
    """Raise ValueError on any broken invariant (see the module docstring for why each matters)."""
    problems = []
    if e.source == "induced":
        if e.recipe is None or e.inject_step is None or e.width is None:
            problems.append("induced entry needs recipe, inject_step and width")
        elif not e.fork_step <= e.inject_step <= e.t0:
            problems.append(f"need fork_step <= inject_step <= t0, got {e.fork_step}, {e.inject_step}, {e.t0}")
    elif e.source == "natural":
        if e.recipe is not None or e.inject_step is not None:
            problems.append("natural entry must not name a recipe or inject_step")
        if e.fork_step > e.t0:
            problems.append(f"need fork_step <= t0, got {e.fork_step} > {e.t0}")
    else:
        problems.append(f"source must be 'induced' or 'natural', got {e.source!r}")
    if e.t0 > e.peak_step or e.lead != e.peak_step - e.t0:
        problems.append(f"need t0 <= peak and lead == peak - t0 (t0={e.t0}, peak={e.peak_step}, lead={e.lead})")
    if e.detection_mode == "predictive" and e.lead < L:
        problems.append(f"predictive detection needs lead >= {L}, got {e.lead}")
    elif e.detection_mode not in ("predictive", "onset"):
        problems.append(f"detection_mode must be 'predictive' or 'onset', got {e.detection_mode!r}")
    if not e.normal_snapshot_uris:
        problems.append("the localizer needs >= 1 normal-step snapshot")
    if e.label not in ("provisional", "confirmed"):
        problems.append(f"label must be 'provisional' or 'confirmed', got {e.label!r}")
    elif e.label == "confirmed" and (not e.evidence or (e.source == "induced" and e.seed == TUNING_SEED)):
        problems.append("'confirmed' needs an evidence file and, for induced spikes, a held-out seed")
    if e.hardware.get("backend") not in ("cuda", "rocm"):
        problems.append(f"hardware.backend must be 'cuda' or 'rocm', got {e.hardware.get('backend')!r}")
    if problems:
        raise ValueError(f"{e.entry_id}: " + "; ".join(problems))


def entries_from_run(run: dict, *, evidence: str, L: int = 2) -> list[SpikeEntry]:
    """Turn one capture run.json into manifest entries -- only triggers that qualify under the run's
    declared policy (induced) or land on a measured natural spike event (natural)."""
    from research.spikes.natural import find_spikes
    from research.spikes.tune_detector import detection_qualifies, spike_occurred

    rec = run.get("recipe")
    common = {
        "config_path": run["config_path"],
        "config_overrides": run["overrides"],
        "data_preset": run["data_preset"],
        "data_revision": run["data_revision"],
        "seed": run["seed"],
        "detector_params": run["detector_params"],
        "normal_snapshot_uris": tuple(run["normal_snapshots"]),
        "evidence": evidence,
        "hardware": run["hardware"],
    }
    out = []
    for trig in run["triggers"]:
        t0 = trig["t0"]
        if rec is not None:
            occ = spike_occurred(run["losses"], rec["inject_step"], peak_offset=rec.get("peak_offset", 0))
            ok, lead = detection_qualifies(
                occurred=occ["occurred"],
                inject_step=rec["inject_step"],
                trigger_step=t0,
                peak_step=occ["peak_step"],
                min_lead=rec["min_lead"],
                allow_at_peak=rec["allow_at_peak"],
            )
            if not ok:
                continue
            ident = {
                "entry_id": f"{rec['kind']}-i{rec['inject_step']}-s{run['seed']}-t{t0}",
                "source": "induced",
                "recipe": rec["kind"],
                "recipe_params": rec.get("params", {}),
                "inject_step": rec["inject_step"],
                "width": rec["width"],
                "peak_step": occ["peak_step"],
                "peak_ratio": occ["ratio"],
                "lead": lead,
                "detection_mode": "onset" if rec["allow_at_peak"] else "predictive",
            }
        else:
            events = find_spikes(run["losses"])
            ev = next((e for e in events if e["start"] - MATCH_BEFORE <= t0 <= e["peak_step"]), None)
            if ev is None:
                continue  # a false alarm in a natural run: counted in run.json, never enters K2
            lead = ev["peak_step"] - t0
            ident = {
                "entry_id": f"natural-{run['stress']}-s{run['seed']}-t{t0}",
                "source": "natural",
                "recipe": None,
                "recipe_params": {"stress": run["stress"]},
                "inject_step": None,
                "width": None,
                "peak_step": ev["peak_step"],
                "peak_ratio": ev["ratio"],
                "lead": lead,
                "detection_mode": "predictive" if lead >= L else "onset",
            }
        out.append(
            SpikeEntry(
                **ident,
                **common,
                t0=t0,
                trigger_stat=trig["stat"],
                t0_snapshot_uri=trig["t0_snapshot"],
                fork_snapshot_uri=trig["fork_snapshot"],
                fork_step=trig["fork_step"],
                vbar_uri=trig["vbar"],
            )
        )
    return out


def build_k2(run_paths: list[str], *, L: int = 2) -> K2Manifest:
    """Collect qualifying spikes from capture runs into a validated manifest. An entry that breaks an
    invariant is left out and reported (one odd run must not sink a whole unattended battery)."""
    entries = []
    for path in run_paths:
        with open(path) as fp:
            for e in entries_from_run(json.load(fp), evidence=path, L=L):
                try:
                    validate(e, L=L)
                    entries.append(e)
                except ValueError as err:
                    print(f"K2: rejected {err}")
    return K2Manifest(entries=entries)


def save_k2(manifest: K2Manifest, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(asdict(manifest), fp, indent=2)


def load_k2(path: str) -> K2Manifest:
    """Read a manifest. Consumers iterate `.entries` and load snapshots via harness.snapshot.load()."""
    with open(path) as fp:
        raw = json.load(fp)
    entries = [SpikeEntry(**{**e, "normal_snapshot_uris": tuple(e["normal_snapshot_uris"])}) for e in raw["entries"]]
    return K2Manifest(version=raw["version"], entries=entries)

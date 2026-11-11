"""
Optimizer Autopsy: paper figures, built against DUMMY data.

Every number below is fake and every figure is stamped "DUMMY DATA".
Swap the make_dummy_*() functions for loaders that read the real results
(see attribution.py's output format) when results land. The plotting
functions only expect the simple shapes documented in each docstring.

Effects are shown in units of the chaos floor (d = delta / sigma_chaos), because
the T1 pilot found most spike effects sit near the rounding-noise level.

Run:  python figures_dummy.py     (writes fig1..fig5 + figG_gate_b PNGs to ./figs)

REAL FORMAT (from research/analysis/attribution.py): gate_b_verdict(deltas, tau=...) takes
    deltas = {recipe: {branch: [delta per matched seed]}},  delta = val_loss(branch) - val_loss(B*)
and returns a dict with "per_recipe", "tau", "decision", "provisional". fig_gate_b() below plots
that dict directly, so with real data you call:
    from research.analysis.attribution import gate_b_verdict
    fig_gate_b(gate_b_verdict(deltas, tau=...), "figs/figG_gate_b.png")
attribute() and necessity_sufficiency() are still TODO there, so figs 1-5 have no real format yet.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUT = "figs"
STAMP = "DUMMY DATA"
rng = np.random.default_rng(0)


def _stamp(ax):
    ax.text(0.5, 0.5, STAMP, transform=ax.transAxes, ha="center", va="center",
            fontsize=34, color="red", alpha=0.12, rotation=25)


# ---------- dummy data (replace with real loaders) ----------
def make_dummy_causal_map():
    """dict: component -> (mean_effect_d, ci_low, ci_high). Effect in chaos-floor units."""
    return {"w": (0.4, -0.3, 1.1), "m": (1.2, 0.4, 2.0), "v": (4.8, 3.9, 5.7),
            "random subspace (control)": (0.3, -0.4, 1.0)}


def make_dummy_psi():
    """1-D array of psi_k, one per spike (fraction of poison mass in top-k subspace)."""
    return np.clip(rng.beta(5, 2, size=60), 0, 1)


def make_dummy_scatter():
    """(predicted_psi, measured_advantage) per spike. Advantage = global-reset loss - selective-repair loss (nats)."""
    psi = rng.uniform(0.2, 1.0, 40)
    adv = 0.02 * (psi - 0.55) * 10 + rng.normal(0, 0.01, 40)
    return psi, adv


def make_dummy_curves(steps=400):
    """dict: method -> (steps, mean_loss_gap, sd). Gap = loss minus clean counterfactual loss."""
    t = np.arange(steps)
    out = {}
    for name, amp, tau in [("no-op (spike)", 0.30, 220), ("skip-step", 0.18, 120),
                           ("global reset", 0.12, 70), ("selective repair", 0.04, 40)]:
        mean = amp * np.exp(-t / tau)
        out[name] = (t, mean, 0.01 + 0.2 * mean)
    return out


def make_dummy_compute():
    """dict: method -> steps needed to re-enter delta-band around the clean path (list over spikes)."""
    return {"no-op": rng.normal(900, 120, 12), "skip-step": rng.normal(520, 90, 12),
            "SPAM": rng.normal(430, 80, 12), "ZClip": rng.normal(480, 85, 12),
            "global reset": rng.normal(350, 60, 12), "selective repair": rng.normal(210, 40, 12)}


# ---------- figures ----------
def fig1_causal_map(data, path):
    names = list(data)
    mean = np.array([data[n][0] for n in names])
    lo = mean - np.array([data[n][1] for n in names])
    hi = np.array([data[n][2] for n in names]) - mean
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(names, mean, yerr=[lo, hi], capsize=4, color=["#4C78A8"] * 3 + ["#9E9E9E"])
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="gray", ls="--", lw=0.8)
    ax.text(len(names) - 0.5, 1.05, "chaos floor (d = 1)", ha="right", fontsize=8, color="gray")
    ax.set_ylabel("Recovery effect (d = delta / sigma_chaos)")
    ax.set_title("Fig 1. Where does the damage live? Causal map")
    plt.xticks(rotation=15, ha="right")
    _stamp(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig2_psi_hist(psi, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(psi, bins=15, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("psi_k (share of poison in top-k curvature subspace)")
    ax.set_ylabel("Number of spikes")
    ax.set_title("Fig 2. How concentrated is the poison?")
    _stamp(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig3_scatter(psi, adv, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(psi, adv, color="#4C78A8")
    ax.axhline(0, color="k", lw=0.8)
    z = np.polyfit(psi, adv, 1)
    xs = np.linspace(psi.min(), psi.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#E45756", lw=1.2)
    ax.set_xlabel("Measured psi_k (theory's predictor)")
    ax.set_ylabel("Advantage of selective repair over global reset (nats)")
    ax.set_title("Fig 3. Does theory predict practice?")
    _stamp(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig4_recovery(curves, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (t, m, sd) in curves.items():
        ax.plot(t, m, label=name)
        ax.fill_between(t, m - sd, m + sd, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Steps after fork")
    ax.set_ylabel("Loss gap to clean counterfactual (nats)")
    ax.set_title("Fig 4. Recovery curves")
    ax.legend(frameon=False)
    _stamp(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig5_compute(data, path):
    names = list(data)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot([data[n] for n in names], tick_labels=names, showmeans=True)
    ax.set_ylabel("Steps to re-enter delta-band of clean path")
    ax.set_title("Fig 5. Compute to recover")
    plt.xticks(rotation=20, ha="right")
    _stamp(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- Gate B figure (uses the REAL verdict format from attribution.py) ----------
def make_dummy_gate_b():
    """Verdict-shaped dict, same keys gate_b_verdict() returns. Replace with the real call."""
    tau = 0.01
    spec = {  # recipe -> branch -> (mean, half-width of CI), in nats
        "lr_bump":       {"B0": (0.060, 0.012), "Bg": (0.020, 0.008), "Bs": (0.030, 0.010), "Bspam": (0.025, 0.009)},
        "corrupt_batch": {"B0": (0.045, 0.010), "Bg": (0.015, 0.007), "Bs": (0.008, 0.006), "Bspam": (0.014, 0.007)},
    }
    per = {}
    for r, bd in spec.items():
        per[r] = {}
        for b, (m, h) in bd.items():
            per[r][b] = {"mean": m, "ci_lo": m - h, "ci_hi": m + h, "n": 5, "recovers": (m + h) <= tau}
    return {"decision": "PROCEED", "provisional": True, "tau": tau,
            "recipes_present": list(spec), "recipes_total": 4, "per_recipe": per}


def fig_gate_b(v, path, stamp=True):
    """Grouped bars: mean delta vs clean counterfactual per branch, with 95% CI and the tau line.
    Hollow-edged bars = branch recovers (CI upper bound <= tau). Bs (cheap fix) is highlighted."""
    recipes = v["recipes_present"]
    branches = []
    for r in recipes:
        for b in v["per_recipe"][r]:
            if b not in branches:
                branches.append(b)
    width = 0.8 / max(len(branches), 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, b in enumerate(branches):
        xs, ms, errs, rec = [], [], [], []
        for i, r in enumerate(recipes):
            s = v["per_recipe"][r].get(b)
            if s is None or not s["n"]:
                continue
            xs.append(i + (j - (len(branches) - 1) / 2) * width)
            ms.append(s["mean"])
            errs.append([s["mean"] - s["ci_lo"], s["ci_hi"] - s["mean"]])
            rec.append(s["recovers"])
        if not xs:
            continue
        colors = "#E45756" if b == "Bs" else "#4C78A8"
        bars = ax.bar(
            xs, ms, width * 0.9, yerr=np.array(errs).T, capsize=3, label=b,
            color=colors, alpha=0.55 + 0.4 * (b == "Bs"),
        )
        for bar, ok in zip(bars, rec):
            if ok:
                bar.set_edgecolor("green")
                bar.set_linewidth(2)
    ax.axhline(v["tau"], color="green", ls="--", lw=1)
    ax.text(len(recipes) - 0.5, v["tau"], " tau (recovery margin)", va="bottom", ha="right", fontsize=8, color="green")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(recipes)))
    ax.set_xticklabels(recipes)
    ax.set_ylabel("Delta = val_loss(branch) - val_loss(B*)  (nats)")
    tag = " (PROVISIONAL)" if v.get("provisional") else ""
    ax.set_title(f"Gate B: cheap-fix kill-test. Decision: {v['decision']}{tag}")
    ax.legend(frameon=False, title="branch (green edge = recovers)", fontsize=8)
    if v.get("provisional"):
        k, tot = len(recipes), v["recipes_total"]
        fig.text(0.5, 0.005, f"PROVISIONAL: {k}/{tot} spike recipes run; not the binding Gate B verdict.",
                 ha="center", fontsize=8, color="darkred")
    if stamp:
        _stamp(ax)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_causal_map(make_dummy_causal_map(), f"{OUT}/fig1_causal_map.png")
    fig2_psi_hist(make_dummy_psi(), f"{OUT}/fig2_psi_hist.png")
    p, a = make_dummy_scatter()
    fig3_scatter(p, a, f"{OUT}/fig3_theory_vs_practice.png")
    fig4_recovery(make_dummy_curves(), f"{OUT}/fig4_recovery_curves.png")
    fig5_compute(make_dummy_compute(), f"{OUT}/fig5_compute_to_recover.png")
    fig_gate_b(make_dummy_gate_b(), f"{OUT}/figG_gate_b.png")
    print("wrote 6 figures to", OUT)

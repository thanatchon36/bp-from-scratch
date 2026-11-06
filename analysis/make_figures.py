"""Build every figure from results/*.json.   python analysis/make_figures.py

Figures are drawn at the printed text width with the shared style in analysis/figstyle.py: PDF
for paper/figures/ and 300 dpi PNG for results/figures/. The headline numbers used by the tables
and the web reader are written to results/summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import figstyle

figstyle.apply()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from common import fit_log2_slope, haar_variance_two_sided, load_all  # noqa: E402

FIG_PDF = ROOT / "paper" / "figures"
FIG_PNG = ROOT / "results" / "figures"
TEXT_IN = figstyle.TEXT_IN
ERR = figstyle.ERR
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")


def savefig(fig, name):
    figstyle.save(fig, FIG_PDF, FIG_PNG, name)
    plt.close(fig)
    print("wrote", name)


def page_entropy(dA: int, dB: int) -> float:
    """Page (1993): mean entanglement entropy (nats) of a Haar state, dA <= dB."""
    if dA > dB:
        dA, dB = dB, dA
    return float(sum(1.0 / k for k in range(dB + 1, dA * dB + 1)) - (dA - 1) / (2.0 * dB))


def kurtosis(x):
    x = np.asarray(x)
    m2 = np.mean(x**2)
    return float(np.mean(x**4) / m2**2) if m2 > 0 else float("nan")


def errbars(var, ci):
    lo, hi = ci
    return np.array([[max(var - lo, 0)], [max(hi - var, 0)]])


def log_ticks(axis, ticks):
    """Plain-number major ticks on a log axis, no minor-tick labels."""
    axis.set_major_formatter(PLAIN)
    axis.set_minor_formatter(NullFormatter())
    axis.set_ticks(ticks)


def hline(ax, y, label=None):
    return ax.axhline(y, color=figstyle.INK, linestyle=(0, (4, 2)), linewidth=0.9, label=label, zorder=1.5)


def series(ax, x, y, color, marker="o", label=None, yerr=None, ls="-"):
    """One data series: thin line, small marker with a white ring, thin error bars."""
    return ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, linestyle=ls, label=label,
                       **figstyle.LINE, **ERR)


# ----------------------------------------------------------------------------- E1
def fig_e1():
    rows = load_all("e1")
    if not rows:
        return {}
    Ls = sorted({r["L"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    ns6 = [n for n in ns if n >= 6]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.35))

    ax = axes[0]
    cols = figstyle.ramp(len(Ls))
    for i, L in enumerate(Ls):
        pts = sorted([r for r in rows if r["L"] == L], key=lambda r: r["n"])
        err = np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) for r in pts])
        series(ax, [r["n"] for r in pts], [r["var_mid"] for r in pts], cols[i], label=f"$L = {L}$", yerr=err)
    figstyle.refline(ax, ns, [haar_variance_two_sided(2**n, 0.0, 2.0**n) for n in ns], label="Haar, two-sided")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_{\mathrm{mid}} C]$")
    figstyle.panel(ax, "a", r"variance vs $n$, $H = Z^{\otimes n}$")
    ax.legend(loc="lower left", labelspacing=0.2)

    ax = axes[1]
    cols = figstyle.ramp(len(ns6))
    for i, n in enumerate(ns6):
        pts = sorted([r for r in rows if r["n"] == n], key=lambda r: r["L"])
        err = np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) / r["haar_two_sided"] for r in pts])
        series(ax, [r["L"] for r in pts], [r["var_mid"] / r["haar_two_sided"] for r in pts], cols[i],
               label=f"$n = {n}$", yerr=err)
    hline(ax, 1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_ticks(ax.xaxis, Ls)
    log_ticks(ax.yaxis, [0.5, 1, 2, 5])
    ax.set_xlabel("depth $L$")
    ax.set_ylabel("variance / Haar value")
    figstyle.panel(ax, "b", "approach to the 2-design value")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[2]
    for i, n in enumerate(ns6):
        pts = sorted([r for r in rows if r["n"] == n], key=lambda r: r["L"])
        series(ax, [r["L"] for r in pts], [kurtosis(r["raw_mid"]) for r in pts], cols[i], label=f"$n = {n}$")
    gauss = hline(ax, 3.0, label="Gaussian (3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_ticks(ax.xaxis, Ls)
    log_ticks(ax.yaxis, [3, 10, 30, 100])
    ax.set_xlabel("depth $L$")
    ax.set_ylabel(r"kurtosis of $\partial_{\mathrm{mid}} C$")
    figstyle.panel(ax, "c", "heavy tails at shallow depth")
    ax.legend([gauss], ["Gaussian (3)"], loc="upper right")
    savefig(fig, "fig_e1")

    slopes = {}
    for L in Ls:
        pts = sorted([r for r in rows if r["L"] == L and r["n"] >= 6], key=lambda r: r["n"])
        s, b = fit_log2_slope([r["n"] for r in pts], [r["var_mid"] for r in pts])
        sm, _ = fit_log2_slope([r["n"] for r in pts], [r["var_mean_over_params"] for r in pts])
        big = max(pts, key=lambda r: r["n"])
        slopes[L] = {"slope_mid": s, "slope_mean": sm, "ratio_to_haar_at_nmax": big["var_mid"] / big["haar_two_sided"],
                     "nmax": big["n"], "kurt_nmax": kurtosis(big["raw_mid"])}
    hs, _ = fit_log2_slope(ns, [haar_variance_two_sided(2**n, 0.0, 2.0**n) for n in ns])
    return {"e1_slopes": slopes, "haar_slope": hs}


# ----------------------------------------------------------------------------- E4
def fig_e4():
    rows = load_all("e4")
    e1 = load_all("e1")
    if not rows:
        return {}
    n = rows[0]["n"]
    pts = sorted(rows, key=lambda r: r["L"])
    page = page_entropy(2 ** (n // 2), 2 ** (n - n // 2))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.5))

    ax = axes[0]
    x = [r["L"] for r in pts]
    err = np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) / r["haar_two_sided"] for r in pts])
    series(ax, x, [r["var_mid"] / r["haar_two_sided"] for r in pts], figstyle.BLUE, label="gradient variance / Haar value", yerr=err)
    series(ax, x, [r["entropy_mean"] / page for r in pts], figstyle.ORANGE, marker="s", label=r"entropy $S_{n/2}$ / Page value")
    hline(ax, 1.0)
    ax.set_xscale("log")
    log_ticks(ax.xaxis, [1, 2, 5, 10, 20, 50, 100])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("depth $L$")
    ax.set_ylabel("ratio to the Haar / Page value")
    figstyle.panel(ax, "a", f"variance and entanglement vs depth, $n = {n}$")
    ax.legend(loc="upper right")

    ax = axes[1]
    if e1:
        Ls = sorted({r["L"] for r in e1})
        ns = sorted({r["n"] for r in e1})
        cols = figstyle.ramp(len(Ls))
        for i, L in enumerate(Ls):
            p2 = sorted([r for r in e1 if r["L"] == L], key=lambda r: r["n"])
            series(ax, [r["n"] for r in p2], [r["entropy_mean"] for r in p2], cols[i], label=f"$L = {L}$")
        figstyle.refline(ax, ns, [page_entropy(2 ** (m // 2), 2 ** (m - m // 2)) for m in ns], label="Page value")
        ax.set_xticks(ns)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"half-chain entropy $S_{n/2}$ (nats)")
    figstyle.panel(ax, "b", r"entanglement of $U(\theta)\,|0\rangle$")
    ax.legend(loc="upper left", ncol=2)
    savefig(fig, "fig_e4")
    deep = max(pts, key=lambda r: r["L"])
    return {"e4_n": n, "e4_page": page, "e4_entropy_deepest": deep["entropy_mean"], "e4_L_deepest": deep["L"],
            "e4_ratio_deepest": deep["var_mid"] / deep["haar_two_sided"]}


# ----------------------------------------------------------------------------- E2
def fig_e2():
    rows = load_all("e2")
    if not rows:
        return {}
    Ls = sorted({r["L"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    cols = figstyle.ramp(len(Ls))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.7), sharey=True)
    out = {}
    titles = {"global": r"global cost $H = |0\rangle\langle 0|^{\otimes n}$",
              "local": r"local cost $H = \frac{1}{2n}\,\Sigma_i\, Z_i$"}
    for ax, letter, cost in zip(axes, "ab", ["global", "local"]):
        for i, L in enumerate(Ls):
            pts = sorted([r for r in rows if r["cost"] == cost and r["L"] == L], key=lambda r: r["n"])
            if not pts:
                continue
            err = np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) for r in pts])
            series(ax, [r["n"] for r in pts], [r["var_mid"] for r in pts], cols[i], label=f"$L = {L}$", yerr=err)
            if len(pts) >= 3:
                s, _ = fit_log2_slope([r["n"] for r in pts if r["n"] >= 6], [r["var_mid"] for r in pts if r["n"] >= 6])
                out[f"e2_slope_{cost}_L{L}"] = s
        if cost == "global":
            haar = [haar_variance_two_sided(2**n, 1.0, 1.0) for n in ns]
        else:
            haar = [haar_variance_two_sided(2**n, 0.0, 2.0**n / (4 * n)) for n in ns]
        figstyle.refline(ax, ns, haar, label="Haar, two-sided")
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_xlabel("qubits $n$")
        figstyle.panel(ax, letter, titles[cost])
    axes[0].set_ylabel(r"$\mathrm{Var}[\partial_{\mathrm{mid}} C]$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    savefig(fig, "fig_e2")
    return out


# ----------------------------------------------------------------------------- E3
def fig_e3():
    rows = load_all("e3")
    if not rows:
        return {}
    gauss = figstyle.ramp(3)
    style = {  # variant -> (label, colour, marker, linestyle)
        "uniform": (r"uniform $[0, 2\pi)$", figstyle.MUTED, "x", (0, (4, 2))),
        "gauss_1.0": (r"Gaussian, $\sigma = 1$", gauss[0], "o", "-"),
        "gauss_0.3": (r"Gaussian, $\sigma = 0.3$", gauss[1], "o", "-"),
        "gauss_0.1": (r"Gaussian, $\sigma = 0.1$", gauss[2], "o", "-"),
        "identity_block": ("identity blocks", figstyle.ORANGE, "D", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.7), sharey=True)
    out = {}
    for ax, letter, which, title in zip(axes, "ab", ["first", "mid"], ["first-layer parameter", "middle-layer parameter"]):
        for v, (label, col, marker, ls) in style.items():
            pts = sorted([r for r in rows if r["variant"] == v], key=lambda r: r["n"])
            if not pts:
                continue
            x = [r["n"] for r in pts]
            y, err = [], []
            for r in pts:
                raw = np.asarray(r[f"raw_{which}"])
                ms = float(np.mean(raw**2))
                boot = np.mean(raw[np.random.default_rng(0).integers(0, len(raw), size=(400, len(raw)))] ** 2, axis=1)
                y.append(ms)
                err.append(errbars(ms, (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))))
            line = series(ax, x, y, col, marker=marker, label=label, yerr=np.hstack(err), ls=ls)
            if marker == "x":
                line[0].set_markeredgecolor(col)
            if len(pts) >= 3:
                s, _ = fit_log2_slope(x, y)
                out[f"e3_slope_{which}_{v}"] = s
        ax.set_yscale("log")
        ax.set_xticks(sorted({r["n"] for r in rows}))
        ax.set_xlabel("qubits $n$")
        figstyle.panel(ax, letter, title)
    axes[0].set_ylabel(r"$\mathbb{E}[(\partial C)^2]$ at initialisation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    savefig(fig, "fig_e3")
    return out


# ----------------------------------------------------------------------------- E5
def fig_e5():
    rows = load_all("e5")
    if not rows:
        return {}
    for r in rows:
        r.setdefault("shots", 0)
    colour = {"global": figstyle.BLUE, "local": figstyle.ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.5))
    out = {}

    ax = axes[0]
    for cost in ["global", "local"]:
        for shots in [0, 10000]:
            pts = sorted([r for r in rows if r["cost"] == cost and r["shots"] == shots], key=lambda r: r["n"])
            if not pts:
                continue
            label = f"{cost} cost, " + ("exact gradients" if shots == 0 else f"$10^4$ shots")
            series(ax, [r["n"] for r in pts], [r["fidelity_mean"] for r in pts], colour[cost],
                   marker="o" if shots == 0 else "s", ls="-" if shots == 0 else (0, (4, 2)),
                   label=label, yerr=[r["fidelity_std"] for r in pts])
            for r in pts:
                out[f"e5_{cost}_shots{shots}_n{r['n']}"] = {
                    "fid": r["fidelity_mean"], "succ": r["success_rate"],
                    "succ90": float(np.mean([u["fidelity_final"] > 0.9 for u in r["runs"]])),
                    "g0": float(np.mean([u["grad_norm_init"] for u in r["runs"]]))}
    ax.set_xticks(sorted({r["n"] for r in rows}))
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"final fidelity $|\langle 0|\psi\rangle|^2$")
    figstyle.panel(ax, "a", r"preparing $|0\rangle^{\otimes n}$ with Adam")
    ax.legend(loc="lower left")

    ax = axes[1]
    exact = {(r["cost"], r["n"], run["seed"]): run["grad_norm_init"]
             for r in rows if r["shots"] == 0 for run in r["runs"]}
    for cost in ["global", "local"]:
        for shots in [0, 10000]:
            pts = [r for r in rows if r["cost"] == cost and r["shots"] == shots]
            if not pts:
                continue
            g0 = [exact.get((cost, r["n"], run["seed"]), run["grad_norm_init"]) for r in pts for run in r["runs"]]
            ff = [run["fidelity_final"] for r in pts for run in r["runs"]]
            col = colour[cost]
            if shots == 0:
                ax.scatter(g0, ff, s=11, marker="o", color=col, edgecolors="white", linewidths=0.4, alpha=0.85,
                           label=f"{cost} cost, exact")
            else:
                ax.scatter(g0, ff, s=14, marker="s", facecolors="none", edgecolors=col, linewidths=0.7, alpha=0.9,
                           label=f"{cost} cost, $10^4$ shots")
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(r"$\|\nabla C\|$ at initialisation (exact)")
    ax.set_ylabel("final fidelity")
    figstyle.panel(ax, "b", "initial gradient norm and outcome")
    ax.legend(loc="center right")
    savefig(fig, "fig_e5")
    return out


# ----------------------------------------------------------------------------- E6
def fig_e6():
    rows = load_all("e6")
    e1 = load_all("e1")
    var = [r for r in rows if "p" in r and "var_gamma1" in r]
    ratio = [r for r in rows if "graphs" in r]
    if not var:
        return {}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.5))
    ax = axes[0]
    ps = sorted({r["p"] for r in var})
    cols = figstyle.ramp(len(ps))
    out = {}
    for i, p in enumerate(ps):
        pts = sorted([r for r in var if r["p"] == p], key=lambda r: r["n"])
        norm = [(1.5 * r["n"]) ** 2 for r in pts]
        err = np.hstack([errbars(r["var_gamma1"], r["var_gamma1_ci"]) / k for r, k in zip(pts, norm)])
        series(ax, [r["n"] for r in pts], [r["var_gamma1"] / k for r, k in zip(pts, norm)], cols[i],
               label=f"QAOA, $p = {p}$", yerr=err)
        if len(pts) >= 3:
            s, _ = fit_log2_slope([r["n"] for r in pts if r["n"] >= 6],
                                  [r["var_gamma1"] / (1.5 * r["n"]) ** 2 for r in pts if r["n"] >= 6])
            out[f"e6_slope_p{p}"] = s
    for L, col in [(5, figstyle.LIGHT), (10, figstyle.INK)]:
        pts = sorted([r for r in e1 if r["L"] == L and r["n"] >= 4], key=lambda r: r["n"])
        if pts:
            ax.plot([r["n"] for r in pts], [r["var_mid"] for r in pts], color=col, marker="x", markeredgecolor=col,
                    linestyle=(0, (4, 2)), linewidth=0.9, label=f"HEA, $L = {L}$ (E1)")
    ax.set_yscale("log")
    ax.set_xticks(sorted({r["n"] for r in var}))
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_{\gamma_1} (C/|E|)]$")
    figstyle.panel(ax, "a", "MaxCut QAOA on random 3-regular graphs")
    ax.legend(loc="center right")

    ax = axes[1]
    if ratio:
        pts = sorted(ratio, key=lambda r: r["p"])
        x = [r["p"] for r in pts]
        series(ax, x, [r["best_mean"] for r in pts], figstyle.BLUE, label="best of 4 starts, mean over 3 graphs")
        series(ax, x, [r["all_mean"] for r in pts], figstyle.MUTED, marker="s", ls=(0, (4, 2)), label="mean over all starts")
        ax.set_xticks(x)
        ax.set_xlabel("depth $p$")
        ax.set_ylabel("approximation ratio")
        ax.set_ylim(0.5, 1.0)
        figstyle.panel(ax, "b", f"QAOA solution quality, $n = {pts[0]['n']}$")
        ax.legend(loc="lower right")
        for r in pts:
            out[f"e6_ratio_p{r['p']}"] = {"best": r["best_mean"], "all": r["all_mean"]}
    savefig(fig, "fig_e6")
    return out


# ----------------------------------------------------------------------------- E7
def fig_e7():
    rows = load_all("e7")
    if not rows:
        return {}
    fams = ["hea_full", "hea_real", "qaoa", "hva_tfim"]
    labels = {"hea_full": r"HEA $\{X_i, Y_i, Z_iZ_{i+1}\}$", "hea_real": r"HEA $\{Y_i, Z_iZ_{i+1}\}$",
              "qaoa": "QAOA MaxCut, 3-regular", "hva_tfim": "HVA TFIM"}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.7))

    ax = axes[0]
    ns_all = sorted({r["n"] for r in rows})
    for i, f in enumerate(fams):
        pts = sorted([r for r in rows if r["family"] == f and not r["capped"]], key=lambda r: r["n"])
        if pts:
            series(ax, [r["n"] for r in pts], [r["dim"] for r in pts], figstyle.CAT[i],
                   marker=figstyle.CAT_MARKERS[i], label=labels[f])
    figstyle.refline(ax, ns_all, [4**n - 1 for n in ns_all], label=r"$\dim\,\mathfrak{su}(2^n) = 4^n - 1$")
    ax.set_yscale("log")
    ax.set_xticks(ns_all)
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"$\dim\,\mathfrak{g}$")
    figstyle.panel(ax, "a", "dynamical Lie algebra dimension")

    ax = axes[1]
    allp, allm = [], []
    for i, f in enumerate(fams):
        pts = [r for r in rows if r["family"] == f and not r["capped"] and np.isfinite(r["predicted_var"])]
        if not pts:
            continue
        p = [r["predicted_var"] for r in pts]
        m = [r["measured_var"] for r in pts]
        err = np.hstack([errbars(r["measured_var"], r["measured_var_ci"]) for r in pts])
        ax.errorbar(p, m, yerr=err, color=figstyle.CAT[i], marker=figstyle.CAT_MARKERS[i], linestyle="none", **ERR)
        allp += p
        allm += m
    lo, hi = min(allp + allm) * 0.5, max(allp + allm) * 2
    figstyle.refline(ax, [lo, hi], [lo, hi])
    ax.annotate("measured = predicted", xy=(hi * 0.55, hi * 0.55), xytext=(-4, 4), textcoords="offset points",
                rotation=45, rotation_mode="anchor", ha="right", va="bottom", fontsize=7, color=figstyle.TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"predicted $\mathcal{P}_\mathfrak{g}(\rho)\,\mathcal{P}_\mathfrak{g}(H)/\dim\mathfrak{g}$")
    ax.set_ylabel(r"measured $\mathrm{Var}[C]$")
    figstyle.panel(ax, "b", "Lie-algebraic prediction vs simulation")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside upper center", ncol=len(labels_))
    savefig(fig, "fig_e7")
    ratios = [r["measured_var"] / r["predicted_var"] for r in rows if not r["capped"] and r["predicted_var"] > 0]
    return {"e7_ratio_median": float(np.median(ratios)), "e7_ratio_min": float(min(ratios)), "e7_ratio_max": float(max(ratios)),
            "e7_rows": [{k: r[k] for k in ["family", "n", "dim", "capped", "purity_rho", "purity_H", "predicted_var",
                                           "measured_var"]} for r in rows]}


def main():
    summary = {}
    for fn in [fig_e1, fig_e4, fig_e2, fig_e3, fig_e5, fig_e6, fig_e7]:
        try:
            summary.update(fn())
        except Exception as exc:  # keep going if one experiment is incomplete
            print(f"{fn.__name__} failed: {exc!r}")
    with open(ROOT / "results" / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print("summary written")


if __name__ == "__main__":
    main()

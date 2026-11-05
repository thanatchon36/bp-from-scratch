"""Build every figure from results/*.json.   python analysis/make_figures.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from common import fit_log2_slope, haar_variance_two_sided, load_all  # noqa: E402

FIG_PDF = ROOT / "paper" / "figures"
FIG_PNG = ROOT / "results" / "figures"
FIG_PDF.mkdir(parents=True, exist_ok=True)
FIG_PNG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "axes.titlesize": 9.5,
                     "lines.markersize": 4, "figure.dpi": 120})
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def savefig(fig, name):
    fig.savefig(FIG_PDF / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_PNG / f"{name}.png", bbox_inches="tight", dpi=160)
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


# ----------------------------------------------------------------------------- E1
def fig_e1():
    rows = load_all("e1")
    if not rows:
        return {}
    Ls = sorted({r["L"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    ax = axes[0]
    for i, L in enumerate(Ls):
        pts = sorted([r for r in rows if r["L"] == L], key=lambda r: r["n"])
        x = [r["n"] for r in pts]
        y = [r["var_mid"] for r in pts]
        err = np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) for r in pts])
        ax.errorbar(x, y, yerr=err, fmt="o-", color=COLORS[i], label=f"L = {L}", capsize=2, lw=1)
    hx = np.array(ns)
    ax.plot(hx, [haar_variance_two_sided(2**n, 0.0, 2.0**n) for n in hx], "k--", lw=1.2, label="Haar (two-sided)")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_{\mathrm{mid}} C]$")
    ax.set_title("(a) variance vs. $n$, $H = Z^{\\otimes n}$")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, n in enumerate([n for n in ns if n >= 6]):
        pts = sorted([r for r in rows if r["n"] == n], key=lambda r: r["L"])
        x = [r["L"] for r in pts]
        y = [r["var_mid"] / r["haar_two_sided"] for r in pts]
        err = np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) / r["haar_two_sided"] for r in pts])
        ax.errorbar(x, y, yerr=err, fmt="o-", color=COLORS[i], label=f"n = {n}", capsize=2, lw=1)
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("depth $L$")
    ax.set_ylabel("variance / Haar value")
    ax.set_title("(b) approach to the 2-design value")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for i, n in enumerate([n for n in ns if n >= 6]):
        pts = sorted([r for r in rows if r["n"] == n], key=lambda r: r["L"])
        ax.plot([r["L"] for r in pts], [kurtosis(r["raw_mid"]) for r in pts], "o-", color=COLORS[i], label=f"n = {n}", lw=1)
    ax.axhline(3.0, color="k", ls="--", lw=1.2, label="Gaussian")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("depth $L$")
    ax.set_ylabel(r"kurtosis of $\partial_{\mathrm{mid}} C$")
    ax.set_title("(c) heavy tails at shallow depth")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    savefig(fig, "fig_e1")

    # slopes table data
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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    x = [r["L"] for r in pts]
    ax.errorbar(x, [r["var_mid"] / r["haar_two_sided"] for r in pts],
                yerr=np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) / r["haar_two_sided"] for r in pts]),
                fmt="o-", color=COLORS[0], label="variance / Haar", capsize=2, lw=1)
    ax.plot(x, [r["entropy_mean"] / page for r in pts], "s-", color=COLORS[3], label=r"$S_{n/2}$ / Page", lw=1)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("depth $L$")
    ax.set_ylabel("ratio to Haar value")
    ax.set_title(f"(a) gradient variance vs. entanglement, $n = {n}$")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    if e1:
        Ls = sorted({r["L"] for r in e1})
        ns = sorted({r["n"] for r in e1})
        for i, L in enumerate(Ls):
            p2 = sorted([r for r in e1 if r["L"] == L], key=lambda r: r["n"])
            ax.plot([r["n"] for r in p2], [r["entropy_mean"] for r in p2], "o-", color=COLORS[i], label=f"L = {L}", lw=1)
        ax.plot(ns, [page_entropy(2 ** (m // 2), 2 ** (m - m // 2)) for m in ns], "k--", lw=1.2, label="Page value")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"half-chain entropy $S_{n/2}$ (nats)")
    ax.set_title("(b) entanglement of $U(\\theta)|0\\rangle$")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    out = {}
    for ax, cost, title in zip(axes, ["global", "local"], ["(a) global cost $|0\\rangle\\langle 0|^{\\otimes n}$",
                                                           "(b) local cost $\\frac{1}{2n}\\sum_i Z_i$"]):
        for i, L in enumerate(Ls):
            pts = sorted([r for r in rows if r["cost"] == cost and r["L"] == L], key=lambda r: r["n"])
            if not pts:
                continue
            x = [r["n"] for r in pts]
            y = [r["var_mid"] for r in pts]
            err = np.hstack([errbars(r["var_mid"], r["var_mid_ci"]) for r in pts])
            ax.errorbar(x, y, yerr=err, fmt="o-", color=COLORS[i], label=f"L = {L}", capsize=2, lw=1)
            if len(pts) >= 3:
                s, _ = fit_log2_slope([r["n"] for r in pts if r["n"] >= 6], [r["var_mid"] for r in pts if r["n"] >= 6])
                out[f"e2_slope_{cost}_L{L}"] = s
        if cost == "global":
            ax.plot(ns, [haar_variance_two_sided(2**n, 1.0, 1.0) for n in ns], "k--", lw=1.2, label="Haar (two-sided)")
        else:
            ax.plot(ns, [haar_variance_two_sided(2**n, 0.0, 2.0**n / (4 * n)) for n in ns], "k--", lw=1.2,
                    label="Haar (two-sided)")
        ax.set_yscale("log")
        ax.set_xlabel("qubits $n$")
        ax.set_title(title)
        ax.legend(ncol=2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\mathrm{Var}[\partial_{\mathrm{mid}} C]$")
    savefig(fig, "fig_e2")
    return out


# ----------------------------------------------------------------------------- E3
def fig_e3():
    rows = load_all("e3")
    if not rows:
        return {}
    variants = ["uniform", "gauss_1.0", "gauss_0.3", "gauss_0.1", "identity_block"]
    labels = {"uniform": "uniform $[0, 2\\pi)$", "gauss_1.0": "Gaussian $\\sigma=1$", "gauss_0.3": "Gaussian $\\sigma=0.3$",
              "gauss_0.1": "Gaussian $\\sigma=0.1$", "identity_block": "identity blocks"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    out = {}
    for ax, which, title in zip(axes, ["first", "mid"], ["(a) first-layer parameter", "(b) middle-layer parameter"]):
        for i, v in enumerate(variants):
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
            ax.errorbar(x, y, yerr=np.hstack(err), fmt="o-", color=COLORS[i], label=labels[v], capsize=2, lw=1)
            if len(pts) >= 3:
                s, _ = fit_log2_slope(x, y)
                out[f"e3_slope_{which}_{v}"] = s
        ax.set_yscale("log")
        ax.set_xlabel("qubits $n$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\mathbb{E}[(\partial C)^2]$ at initialisation")
    axes[0].legend()
    savefig(fig, "fig_e3")
    return out


# ----------------------------------------------------------------------------- E5
def fig_e5():
    rows = load_all("e5")
    if not rows:
        return {}
    for r in rows:
        r.setdefault("shots", 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    out = {}
    styles = {(cost, shots): (COLORS[i], "-" if shots == 0 else "--", "o" if shots == 0 else "s")
              for i, cost in enumerate(["global", "local"]) for shots in [0, 10000]}
    for (cost, shots), (col, ls, mk) in styles.items():
        pts = sorted([r for r in rows if r["cost"] == cost and r["shots"] == shots], key=lambda r: r["n"])
        if not pts:
            continue
        x = [r["n"] for r in pts]
        lab = f"{cost} cost, " + ("exact gradients" if shots == 0 else f"{shots} shots")
        ax.errorbar(x, [r["fidelity_mean"] for r in pts], yerr=[r["fidelity_std"] for r in pts], fmt=mk + ls,
                    color=col, capsize=2, lw=1, label=lab)
        for r in pts:
            out[f"e5_{cost}_shots{shots}_n{r['n']}"] = {
                "fid": r["fidelity_mean"], "succ": r["success_rate"],
                "succ90": float(np.mean([u["fidelity_final"] > 0.9 for u in r["runs"]])),
                "g0": float(np.mean([u["grad_norm_init"] for u in r["runs"]]))}
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"final fidelity $|\langle 0|\psi\rangle|^2$")
    ax.set_title("(a) preparing $|0\\rangle^{\\otimes n}$ with Adam")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for (cost, shots), (col, ls, mk) in styles.items():
        pts = [r for r in rows if r["cost"] == cost and r["shots"] == shots]
        if not pts:
            continue
        exact = {(r["cost"], r["n"], run["seed"]): run["grad_norm_init"]
                 for r in rows if r["shots"] == 0 for run in r["runs"]}
        g0 = [exact.get((cost, r["n"], run["seed"]), run["grad_norm_init"]) for r in pts for run in r["runs"]]
        ff = [run["fidelity_final"] for r in pts for run in r["runs"]]
        ax.scatter(g0, ff, s=10, color=col, marker=mk, alpha=0.6, label=f"{cost}, " + ("exact" if shots == 0 else "shots"))
    ax.set_xscale("log")
    ax.set_xlabel(r"$\|\nabla C\|$ at initialisation (exact)")
    ax.set_ylabel("final fidelity")
    ax.set_title("(b) initial gradient norm vs. outcome")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)
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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    ps = sorted({r["p"] for r in var})
    out = {}
    for i, p in enumerate(ps):
        pts = sorted([r for r in var if r["p"] == p], key=lambda r: r["n"])
        x = [r["n"] for r in pts]
        norm = [(1.5 * r["n"]) ** 2 for r in pts]
        y = [r["var_gamma1"] / k for r, k in zip(pts, norm)]
        err = np.hstack([errbars(r["var_gamma1"], r["var_gamma1_ci"]) / k for r, k in zip(pts, norm)])
        ax.errorbar(x, y, yerr=err, fmt="o-", color=COLORS[i], label=f"QAOA p = {p}", capsize=2, lw=1)
        if len(pts) >= 3:
            s, _ = fit_log2_slope([r["n"] for r in pts if r["n"] >= 6],
                                  [r["var_gamma1"] / (1.5 * r["n"]) ** 2 for r in pts if r["n"] >= 6])
            out[f"e6_slope_p{p}"] = s
    for j, L in enumerate([5, 10]):
        pts = sorted([r for r in e1 if r["L"] == L and r["n"] >= 4], key=lambda r: r["n"])
        if pts:
            ax.plot([r["n"] for r in pts], [r["var_mid"] for r in pts], "x--", color="gray" if j == 0 else "k", lw=1,
                    label=f"HEA L = {L} (E1)")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"$\mathrm{Var}[\partial_{\gamma_1} (C/|E|)]$")
    ax.set_title("(a) MaxCut QAOA on random 3-regular graphs")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    ax = axes[1]
    if ratio:
        pts = sorted(ratio, key=lambda r: r["p"])
        x = [r["p"] for r in pts]
        ax.plot(x, [r["best_mean"] for r in pts], "o-", color=COLORS[0], label="best of 4 starts (mean over 3 graphs)")
        ax.plot(x, [r["all_mean"] for r in pts], "s--", color=COLORS[1], label="mean over all starts")
        ax.set_xlabel("depth $p$")
        ax.set_ylabel("approximation ratio")
        ax.set_title(f"(b) QAOA quality, $n = {pts[0]['n']}$")
        ax.set_ylim(0.5, 1.02)
        ax.legend()
        ax.grid(alpha=0.3)
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
    labels = {"hea_full": "HEA $\\{X_i, Y_i, Z_iZ_{i+1}\\}$", "hea_real": "HEA $\\{Y_i, Z_iZ_{i+1}\\}$",
              "qaoa": "QAOA MaxCut (3-regular)", "hva_tfim": "HVA TFIM"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    ns_all = sorted({r["n"] for r in rows})
    for i, f in enumerate(fams):
        pts = sorted([r for r in rows if r["family"] == f and not r["capped"]], key=lambda r: r["n"])
        if pts:
            ax.plot([r["n"] for r in pts], [r["dim"] for r in pts], "o-", color=COLORS[i], label=labels[f], lw=1)
    ax.plot(ns_all, [4**n - 1 for n in ns_all], "k--", lw=1, label="$\\dim\\,\\mathfrak{su}(2^n) = 4^n - 1$")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel(r"$\dim\,\mathfrak{g}$")
    ax.set_title("(a) dynamical Lie algebra dimension")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    allp, allm = [], []
    for i, f in enumerate(fams):
        pts = [r for r in rows if r["family"] == f and not r["capped"] and np.isfinite(r["predicted_var"])]
        if not pts:
            continue
        p = [r["predicted_var"] for r in pts]
        m = [r["measured_var"] for r in pts]
        err = np.hstack([errbars(r["measured_var"], r["measured_var_ci"]) for r in pts])
        ax.errorbar(p, m, yerr=err, fmt="o", color=COLORS[i], label=labels[f], capsize=2)
        allp += p
        allm += m
    lo, hi = min(allp + allm) * 0.5, max(allp + allm) * 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"predicted $P_\mathfrak{g}(\rho)P_\mathfrak{g}(H)/\dim\mathfrak{g}$")
    ax.set_ylabel(r"measured $\mathrm{Var}[C]$")
    ax.set_title("(b) Lie-algebraic prediction vs. simulation")
    ax.grid(alpha=0.3)
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

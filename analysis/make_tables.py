"""Write LaTeX tables into paper/tables/ from results/*.json.   python analysis/make_tables.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from common import load_all  # noqa: E402

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def sci(x):
    if x == 0 or not np.isfinite(x):
        return "--"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    return f"${m:.2f}\\times10^{{{e}}}$"


def dec2(x):
    """Two significant digits as a plain decimal (0.164 -> 0.16, 0.000148 -> 0.00015)."""
    if x == 0 or not np.isfinite(x):
        return "--"
    return np.format_float_positional(x, precision=2, unique=False, fractional=False, trim="-")


def table_haar():
    rows = load_all("e0")
    if not rows:
        return
    lines = [r"\begin{tabular}{ccccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{two-sided} & \multicolumn{2}{c}{after only ($U_R = \mathbb{1}$)} & \multicolumn{2}{c}{before only ($U_L = \mathbb{1}$)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"$n$ & simulated & Eq.~\eqref{eq:haar2} & simulated & Eq.~\eqref{eq:haarL} & simulated & Eq.~\eqref{eq:haarR} \\",
             r"\midrule", r"\multicolumn{7}{l}{$H = Z^{\otimes n}$} \\"]
    for r in rows:
        g = r["globalZ"]
        lines.append(f"{r['n']} & {sci(g['measured']['two_sided'])} & {sci(g['predicted']['two_sided'])} & "
                     f"{sci(g['measured']['left_only'])} & {sci(g['predicted']['left_only'])} & "
                     f"{sci(g['measured']['right_only'])} & {sci(g['predicted']['right_only'])} \\\\")
    lines += [r"\midrule", r"\multicolumn{7}{l}{$H = |0\rangle\langle 0|^{\otimes n}$} \\"]
    for r in rows:
        g = r["projector"]
        lines.append(f"{r['n']} & {sci(g['measured']['two_sided'])} & {sci(g['predicted']['two_sided'])} & "
                     f"{sci(g['measured']['left_only'])} & {sci(g['predicted']['left_only'])} & "
                     f"{sci(g['measured']['right_only'])} & {sci(g['predicted']['right_only'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "haar_check.tex").write_text("\n".join(lines) + "\n")


def table_slopes():
    with open(ROOT / "results" / "summary.json") as fh:
        s = json.load(fh)
    sl = s.get("e1_slopes")
    if not sl:
        return
    e1 = load_all("e1")
    at = f"at $n = {max(r['n'] for r in e1)}$" if e1 else r"at $n_{\max}$"
    lines = [r"\begin{tabular}{ccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{decay exponent of $\log_2\Var$} & \multicolumn{2}{c}{" + at + r"} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"depth $L$ & $\partial_{\rm mid}C$ & mean over parameters & Var/Haar & kurtosis \\",
             r"\midrule"]
    for L in sorted(sl, key=int):
        v = sl[L]
        lines.append(f"{L} & ${v['slope_mid']:.2f}$ & ${v['slope_mean']:.2f}$ & ${v['ratio_to_haar_at_nmax']:.2f}$ "
                     f"& ${v['kurt_nmax']:.1f}$ \\\\")
    lines += [r"\midrule", f"Haar (two-sided) & ${s['haar_slope']:.2f}$ & -- & 1 & 3 \\\\", r"\bottomrule", r"\end{tabular}"]
    (OUT / "slopes.tex").write_text("\n".join(lines) + "\n")


def table_dla():
    rows = load_all("e7")
    if not rows:
        return
    fams = [("hea_full", r"HEA $\{X_i,Y_i,Z_iZ_{i+1}\}$"), ("hea_real", r"HEA $\{Y_i,Z_iZ_{i+1}\}$"),
            ("qaoa", "QAOA MaxCut"), ("hva_tfim", "HVA TFIM")]
    ns = sorted({r["n"] for r in rows})
    ncol = len(ns) + 1
    lines = [r"\begin{tabular}{l" + "c" * len(ns) + "}", r"\toprule",
             "family / $n$ & " + " & ".join(str(n) for n in ns) + r" \\", r"\midrule",
             rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{DLA dimension}}}} \\",
             r"\quad $4^n-1$ & " + " & ".join(str(4**n - 1) for n in ns) + r" \\"]
    for f, lab in fams:
        cells = []
        for n in ns:
            r = [x for x in rows if x["family"] == f and x["n"] == n]
            cells.append("--" if not r else (f"$>{r[0]['dim']}$" if r[0]["capped"] else str(r[0]["dim"])))
        lines.append(r"\quad " + lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{measured / predicted $\Var[C]$}}}} \\"]
    for f, lab in fams:
        cells = []
        for n in ns:
            r = [x for x in rows if x["family"] == f and x["n"] == n and not x["capped"]]
            cells.append("--" if not r else f"{r[0]['measured_var'] / r[0]['predicted_var']:.2f}")
        lines.append(r"\quad " + lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "dla.tex").write_text("\n".join(lines) + "\n")


def table_e5():
    rows = load_all("e5")
    if not rows:
        return
    for r in rows:
        r.setdefault("shots", 0)
    ns = sorted({r["n"] for r in rows})
    ncol = len(ns) + 1
    lines = [r"\begin{tabular}{l" + "c" * len(ns) + "}", r"\toprule",
             "$n$ & " + " & ".join(str(n) for n in ns) + r" \\", r"\midrule"]
    for shots in [0, 10000]:
        for cost in ["global", "local"]:
            if cost == "local":
                lines.append(r"\addlinespace")
            fid, succ, g0 = [], [], []
            for n in ns:
                r = [x for x in rows if x["cost"] == cost and x["n"] == n and x["shots"] == shots]
                if r:
                    fid.append(f"{r[0]['fidelity_mean']:.3f}")
                    succ.append(f"{np.mean([u['fidelity_final'] > 0.9 for u in r[0]['runs']]):.2f}")
                    ex = [x for x in rows if x["cost"] == cost and x["n"] == n and x["shots"] == 0]
                    g0.append(dec2(float(np.mean([u["grad_norm_init"] for u in ex[0]["runs"]]))) if ex else "--")
                else:
                    fid.append("--")
                    succ.append("--")
                    g0.append("--")
            tag = "exact gradients" if shots == 0 else "$10^4$ shots"
            lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{cost} cost, {tag}}}}} \\")
            lines.append(r"\quad mean fidelity & " + " & ".join(fid) + r" \\")
            lines.append(r"\quad fraction with $F>0.9$ & " + " & ".join(succ) + r" \\")
            if shots == 0:
                lines.append(r"\quad mean $\|\nabla C\|$ at initialisation & " + " & ".join(g0) + r" \\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "training.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    table_haar()
    table_slopes()
    table_dla()
    table_e5()
    print("tables written to", OUT)

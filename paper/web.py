"""Build the web reader docs/paper/ from paper/main.tex (same text as the PDF).

A standalone page built by pandoc with paper/reader.html: masthead, abstract, a results strip whose
numbers come from results/summary.json, contents, numbered sections, [n] citations linked to the
references. The PDF and the PNG figures are copied next to it, and mkdocs serves the folder at
/paper/. Requires pandoc on PATH. Usage: python3 web.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys

from texprep import (
    HERE,
    abstract,
    fix_refs_html,
    main_sections,
    polish_html,
    prepare,
    read_main,
    title_author,
)

ROOT = HERE.parent
OUT = ROOT / "docs" / "paper"
REPO_URL = "https://github.com/thanatchon36/bp-from-scratch"
PAPER_URL = "https://thanatchon36.github.io/bp-from-scratch/paper/"


def num(x: float, digits: int = 2) -> str:
    """Typographic minus, fixed decimals."""
    return f"{x:.{digits}f}".replace("-", "\u2212")


def results() -> list[tuple[str, str, str]]:
    s = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    e1 = s["e1_slopes"]
    e1_slopes = [v["slope_mid"] for v in e1.values()]
    kurt = max(v["kurt_nmax"] for v in e1.values())
    glob = [v for k, v in s.items() if k.startswith("e2_slope_global_")]
    loc = [v for k, v in s.items() if k.startswith("e2_slope_local_")]
    qaoa = [v for k, v in s.items() if k.startswith("e6_slope_p")]
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    return [
        (
            "E1",
            "Gradient variance decays as 2\u207b\u207f at every depth",
            f"fitted exponents {num(min(e1_slopes))} to {num(max(e1_slopes))} vs "
            f"{num(s['haar_slope'])} for a Haar 2-design",
        ),
        (
            "E1",
            f"Kurtosis up to {kurt:.0f} at shallow depth",
            "the derivative becomes Gaussian (kurtosis 3) only where the variance saturates",
        ),
        (
            "E4",
            "Variance saturates before entanglement does",
            f"n = {s['e4_n']}: half-chain entropy reaches {pct(s['e4_ratio_deepest'])} of the "
            f"Page value only at L = {s['e4_L_deepest']}",
        ),
        (
            "E2",
            f"Global-cost exponents {num(min(glob), 1)} to {num(max(glob), 1)}",
            f"local cost {num(min(loc))} to {num(max(loc))}: the cost-function dependence of Cerezo et al.",
        ),
        (
            "E3",
            "Small-angle initialisation flattens the decay",
            f"first-layer exponent {num(s['e3_slope_first_uniform'])} (uniform) vs "
            f"{num(s['e3_slope_first_gauss_0.1'])} (\u03c3 = 0.1)",
        ),
        (
            "E5",
            f"Exact-gradient Adam trains the global cost at n = 14 in "
            f"{pct(s['e5_global_shots0_n14']['succ'])} of runs",
            f"with 10\u2074-shot gradient noise: {pct(s['e5_global_shots10000_n14']['succ'])} at n = 14",
        ),
        (
            "E6",
            "QAOA shows no plateau at fixed depth",
            f"variance exponents {num(min(qaoa))} to {num(max(qaoa))} for p = 1\u20138",
        ),
        (
            "E7",
            f"Lie-algebraic variance formula: median measured/predicted {num(s['e7_ratio_median'])}",
            "{Y\u1d62, Z\u1d62Z\u1d62\u208a\u2081} closes into so(2n), dimension n(2n \u2212 1)",
        ),
    ]


def pandoc(tex: str, args: list[str]) -> str:
    tmp = HERE / ".web.tex"
    tmp.write_text(tex, encoding="utf-8")
    try:
        return subprocess.run(
            ["pandoc", tmp.name, "--from=latex", "--to=html5", "--mathml", "--wrap=none", *args],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    tex = read_main()
    title, subtitle, author = title_author(tex)
    preamble = tex.split("\\begin{document}")[0]
    abstract_html = pandoc(
        preamble + "\\begin{document}\n" + prepare_snippet(abstract(tex)) + "\n\\end{document}\n", []
    )
    page = pandoc(
        prepare(tex, keep_abstract=False),
        [
            f"--template={HERE / 'reader.html'}",
            "--standalone",
            "--number-sections",
            "--toc",
            "--toc-depth=1",
            f"--metadata=title:{title}",
            f"--metadata=subtitle:{subtitle}",
            f"--metadata=author:{author}",
            "--metadata=pagetitle:Barren Plateaus from Scratch",
            "--metadata=description:Technical report: a from-scratch NumPy state-vector study "
            "of barren plateaus in variational quantum circuits.",
        ],
    )
    page = re.sub(
        r'<embed src="figures/([\w-]+)\.pdf"[^>]*/>', r'<img src="figures/\1.png" alt="" />', page
    )
    page = polish_html(fix_refs_html(page), main_sections(tex))
    items = "\n".join(
        f'<li class="r r-ref"><span class="r-id">{exp}</span>'
        f'<span class="r-text">{html.escape(text)}<small>{html.escape(detail)}</small></span></li>'
        for exp, text, detail in results()
    )
    version = re.search(
        r'^__version__ = "([^"]+)"', (ROOT / "qsim" / "__init__.py").read_text(encoding="utf-8"), re.M
    )
    if version is None:
        raise SystemExit("no __version__ in qsim/__init__.py")
    meta = (
        f"<b>bp-from-scratch v{version.group(1)}</b> · NumPy state-vector simulator · "
        "2\u201316 qubits · experiments E0\u2013E7 · every point cached in <code>results/</code>"
    )
    actions = (
        '<li><a href="paper.pdf">Download PDF</a></li>'
        f'<li><a href="{REPO_URL}">Source code</a></li>'
        f'<li><a href="{REPO_URL}#reproduce">Reproduce</a></li>'
        f'<li><a href="{REPO_URL}/blob/main/paper/paper.md">Markdown mirror</a></li>'
    )
    bib = (
        "@techreport{chongmankhong2026barren,\n"
        "  author = {Chongmankhong, Thanatchon},\n"
        f"  title  = {{{title}: {subtitle}}},\n"
        "  year   = {2026},\n"
        "  type   = {Technical report},\n"
        f"  url    = {{{PAPER_URL}}}\n"
        "}"
    )
    cite = f"<h2>How to cite</h2><pre><code>{html.escape(bib)}</code></pre>"
    colophon = (
        "<p>Rendered from <code>paper/main.tex</code> by <code>make paper</code>. Every figure and "
        "table is regenerated from <code>results/*.json</code> by <code>analysis/</code>, and the "
        "headline numbers above come from <code>results/summary.json</code>; the PDF is "
        "authoritative.</p>"
    )
    for marker, value in (
        ("<!--ABSTRACT-->", abstract_html),
        ("<!--META-->", meta),
        ("<!--ACTIONS-->", actions),
        ("<!--RESULTS-NOTE-->", "Headline numbers, generated from the results files."),
        ("<!--RESULTS-->", items),
        ("<!--CITE-->", cite),
        ("<!--COLOPHON-->", colophon),
    ):
        if marker not in page:
            raise SystemExit(f"reader.html lacks {marker}")
        page = page.replace(marker, value)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    shutil.copyfile(HERE / "paper.pdf", OUT / "paper.pdf")
    (OUT / "figures").mkdir(exist_ok=True)
    for name in sorted(set(re.findall(r'src="figures/([\w.-]+)"', page))):
        shutil.copyfile(ROOT / "results" / "figures" / name, OUT / "figures" / name)
    print(f"wrote {OUT / 'index.html'}")
    return 0


def prepare_snippet(text: str) -> str:
    """Apply the inline rewrites (math, citations) to a fragment such as the abstract."""
    return re.sub(r"\{\\rm\s+([^{}]*)\}", r"\\mathrm{\1}", text)


if __name__ == "__main__":
    sys.exit(main())

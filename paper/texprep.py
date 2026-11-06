"""Shared LaTeX preprocessing for the pandoc outputs of paper/main.tex (web reader and Markdown mirror).

pandoc reads main.tex directly but mishandles three constructs that this paper uses: `{\\rm x}` inside
math (not converted), the inline `thebibliography` (all entries merged into one paragraph) and
`\\cite` (empty spans without a .bib file). This module rewrites them into equivalents pandoc
understands: `\\mathrm{x}`, a numbered list with `ref-<key>` anchors, and `[n]` links that follow the
PDF's numbering (order of the `\\bibitem`s). The generated tables are inlined first, so the same
rewrites reach them, and the booktabs commands pandoc cannot render are dropped from them.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIB = re.compile(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", re.S)
ABSTRACT = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)
INPUT = re.compile(r"\\input\{([^}]+)\}")
# \cmidrule leaks its arguments into the first cell and \addlinespace swallows the row after it.
BOOKTABS = re.compile(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}|\\addlinespace(?:\[[^\]]*\])?")


def read_main() -> str:
    return (HERE / "main.tex").read_text(encoding="utf-8")


def title_author(tex: str) -> tuple[str, str, str]:
    """(title, subtitle, author) from \\title{Main:\\\\ \\large Sub} and \\author{Name\\\\ ...}."""
    title = re.search(r"\\title\{(.*?)\}\n", tex, re.S)
    author = re.search(r"\\author\{(.*?)\}\n", tex, re.S)
    main, _, sub = (title.group(1) if title else "Paper").partition("\\\\")
    sub = re.sub(r"\\(large|Large|small|normalsize)\s*", "", sub).strip()
    name = re.sub(r"\\thanks\{.*\}", "", author.group(1) if author else "", flags=re.S)
    name = name.split("\\\\")[0].strip()
    return main.strip().rstrip(":"), sub, name


def abstract(tex: str) -> str:
    m = ABSTRACT.search(tex)
    if m is None:
        raise SystemExit("no abstract in paper/main.tex")
    return m.group(1).strip()


def bib_keys(tex: str) -> list[tuple[str, str]]:
    m = BIB.search(tex)
    if m is None:
        raise SystemExit("no thebibliography in paper/main.tex")
    return re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)", m.group(1), re.S)


def inline_inputs(tex: str) -> str:
    """Replace \\input{file} by the file (the generated tables), minus the booktabs rules pandoc
    cannot render; the \\quad that indents grouped rows becomes two non-breaking spaces."""

    def sub(m: re.Match[str]) -> str:
        path = HERE / m.group(1)
        if not path.suffix:
            path = path.with_suffix(".tex")
        body = BOOKTABS.sub("", path.read_text(encoding="utf-8"))
        return body.replace("\\quad ", "~~")

    return INPUT.sub(sub, tex)


def prepare(tex: str, *, keep_abstract: bool) -> str:
    """Return main.tex rewritten for pandoc. The title block is dropped (the callers render it);
    the abstract becomes an unnumbered section or is removed."""
    tex = BOOKTABS.sub("", inline_inputs(tex))
    entries = bib_keys(tex)
    number = {key: i for i, (key, _) in enumerate(entries, start=1)}

    def cite(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        missing = [k for k in keys if k not in number]
        if missing:
            raise SystemExit(f"\\cite of unknown key(s): {missing}")
        return "[" + ", ".join(f"\\hyperlink{{ref-{k}}}{{{number[k]}}}" for k in keys) + "]"

    refs = "\n".join(f"\\item \\hypertarget{{ref-{k}}}{{}}{body.strip()}" for k, body in entries)
    tex = BIB.sub(lambda _: "\\section*{References}\n\\begin{enumerate}\n" + refs + "\n\\end{enumerate}", tex)
    tex = re.sub(r"\\cite\{([^}]+)\}", cite, tex)
    tex = number_equations(tex)
    tex = re.sub(r"\\paragraph\{", r"\\paragraph*{", tex)
    tex = re.sub(r"\{\\rm\s+([^{}]*)\}", r"\\mathrm{\1}", tex)
    tex = tex.replace("\\maketitle", "")
    tex = ABSTRACT.sub(
        (lambda m: "\\section*{Abstract}\n" + m.group(1)) if keep_abstract else (lambda _: ""), tex
    )
    return tex


def fix_refs_html(html: str) -> str:
    """Move the `ref-<key>` anchor from pandoc's empty <div> onto the list item."""
    return re.sub(r'<li><div id="(ref-[^"]+)">\s*</div>\s*', r'<li id="\1">', html)


def fix_refs_gfm(md: str) -> str:
    return re.sub(r'<div id="(ref-[^"]+)">\s*</div>\s*', r'<a id="\1"></a>', md)


EQ_ENV = re.compile(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", re.S)


def number_equations(tex: str) -> str:
    """Number `equation` and `align` rows as the PDF does (pandoc's MathML drops the numbers),
    anchor each label, and turn \\eqref{k} into a linked (n)."""
    labels: dict[str, int] = {}
    counter = 0

    def env(m: re.Match[str]) -> str:
        nonlocal counter
        name, body = m.group(1), m.group(2)
        rows = re.split(r"\\\\\s*\n", body) if name == "align" else [body]
        anchors, out = "", []
        for row in rows:
            if "\\nonumber" in row or "\\notag" in row:
                out.append(row)
                continue
            counter += 1
            lab = re.search(r"\\label\{([^}]+)\}", row)
            row = re.sub(r"\\label\{[^}]+\}", "", row).rstrip()
            if lab:
                labels[lab.group(1)] = counter
                anchors += f"\\hypertarget{{{lab.group(1)}}}{{}}"
            out.append(row + f"\\qquad\\text{{({counter})}}")
        return anchors + f"\\begin{{{name}}}" + "\\\\\n".join(out) + f"\n\\end{{{name}}}"

    tex = EQ_ENV.sub(env, tex)

    def eqref(m: re.Match[str]) -> str:
        if m.group(1) not in labels:
            raise SystemExit(f"\\eqref of unknown label {m.group(1)}")
        return f"(\\hyperlink{{{m.group(1)}}}{{{labels[m.group(1)]}}})"

    return re.sub(r"\\eqref\{([^}]+)\}", eqref, tex)


def main_sections(tex: str) -> int:
    """Number of numbered top-level sections before \\appendix (pandoc numbers appendices on)."""
    return len(re.findall(r"\\section\{", tex.split("\\appendix")[0]))


def letter(number: str, n_main: int) -> str:
    head, _, rest = number.partition(".")
    if int(head) <= n_main:
        return number
    return chr(ord("A") + int(head) - n_main - 1) + ("." + rest if rest else "")


def polish_html(html: str, n_main: int) -> str:
    """Appendix letters (A, B, ...) and 'Figure n.' / 'Table n.' caption labels."""
    html = re.sub(
        r'(<span class="(?:header|toc)-section-number">)([\d.]+)(</span>)',
        lambda m: m.group(1) + letter(m.group(2), n_main) + m.group(3),
        html,
    )
    html = re.sub(
        r'(data-reference="app:[^"]*">)([\d.]+)(</a>)',
        lambda m: m.group(1) + letter(m.group(2), n_main) + m.group(3),
        html,
    )
    for tag, word in (("figcaption", "Figure"), ("caption", "Table")):
        k = 0

        def label(m: re.Match[str]) -> str:
            nonlocal k
            k += 1
            return f'{m.group(0)}<span class="cap-label">{word} {k}.</span> '

        html = re.sub(rf"<{tag}>", label, html)
    return html


def polish_gfm(md: str, n_main: int) -> str:
    return re.sub(
        r"\[([\d.]+)\]\(#app:", lambda m: f"[{letter(m.group(1), n_main)}](#app:", md
    )

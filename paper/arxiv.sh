#!/usr/bin/env bash
# Package the LaTeX source for arXiv: dist/arxiv/bp-from-scratch-arxiv.tar.gz holds main.tex, the PDF
# figures and the generated tables (the bibliography is inline, so no .bib/.bbl is needed).
# With Docker available, the tarball is also compiled with pdfLaTeX from TeX Live, as arXiv does.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../dist/arxiv
rm -rf "$OUT"
mkdir -p "$OUT/src/figures" "$OUT/src/tables"
cp main.tex "$OUT/src/"
cp figures/*.pdf "$OUT/src/figures/"
cp tables/*.tex "$OUT/src/tables/"
tar -czf "$OUT/bp-from-scratch-arxiv.tar.gz" -C "$OUT/src" .
echo "wrote $(cd "$OUT" && pwd)/bp-from-scratch-arxiv.tar.gz"
if [ "${ARXIV_TEST:-1}" = "1" ] && command -v docker >/dev/null; then
  TEST="$OUT/test"
  mkdir -p "$TEST" && tar -xzf "$OUT/bp-from-scratch-arxiv.tar.gz" -C "$TEST"
  docker run --rm --network none -u "$(id -u):$(id -g)" -v "$(cd "$TEST" && pwd)":/w -w /w \
    "${TEXLIVE_IMAGE:-texlive/texlive:latest-small}" \
    sh -c 'pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null; s=$?; grep -E "^(!|LaTeX Warning: (Reference|Citation)|Output written)" main.log; exit $s'
  echo "pdfLaTeX test build: $TEST/main.pdf"
fi

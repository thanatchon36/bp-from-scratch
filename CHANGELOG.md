# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-06

### Added
- Web reader of the paper (`docs/paper/`, served at https://thanatchon36.github.io/bp-from-scratch/paper/) and a
  Markdown mirror (`paper/paper.md`), both generated from `paper/main.tex` by `make paper`.
- arXiv packaging (`make arxiv`), tested with pdfLaTeX from TeX Live, and the submission guide `paper/ARXIV.md`.
- MkDocs site and GitHub Pages workflow, `CITATION.cff` with a preferred citation, and Zenodo metadata.

### Changed
- Figures redrawn for print with a shared style (`analysis/figstyle.py`): drawn at the final 6.5 in text width
  in 8 pt STIX with embedded Type 42 fonts, a single-hue blue ramp for ordered series (depth, qubits, QAOA depth,
  initialisation width) and fixed categorical hues for the ansatz families, both checked for colour-vision
  deficiency, theory references as black dashed lines, bold panel letters, and legends kept off the data.
- Tables use booktabs grouped headers and indented row groups instead of long row labels, and the E5 gradient
  norms are printed as decimals. No table, equation or code line overruns the text width any more.
- The web reader and Markdown mirror inline the generated tables before pandoc runs, so equation references
  inside tables are numbered and booktabs group headers render.
- The paper names its author. Every reference has been checked against Crossref or the arXiv API, and each
  now carries its DOI or arXiv identifier.
- `make paper` builds with Tectonic (`paper/build.sh`); the CI workflow is now `ci`, with pinned actions.

## [0.1.0] - 2026-11-05

### Added
- `qsim`: NumPy state-vector simulator with parameter-shift, adjoint and finite-difference gradients, Pauli-sum
  costs, entanglement entropies, Haar-random unitaries and exact dynamical Lie algebras.
- Ansätze (McClean HEA, ALA, identity blocks, Pauli-rotation HEA, QAOA MaxCut, HVA TFIM) and experiments E0–E7
  with cached results for every data point.
- Figure and table generation, and the technical report *Barren Plateaus from Scratch*.

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Web reader of the paper (`docs/paper/`, served at https://thanatchon36.github.io/bp-from-scratch/paper/) and a
  Markdown mirror (`paper/paper.md`), both generated from `paper/main.tex` by `make paper`.
- arXiv packaging (`make arxiv`), tested with pdfLaTeX from TeX Live, and the submission guide `paper/ARXIV.md`.
- MkDocs site and GitHub Pages workflow, `CITATION.cff` with a preferred citation, and Zenodo metadata.

### Changed
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

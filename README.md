<p align="center"><img src="docs/assets/logo.svg" alt="bp-from-scratch logo" width="120"></p>

<h1 align="center">bp-from-scratch</h1>

[![CI](https://github.com/thanatchon36/bp-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/thanatchon36/bp-from-scratch/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/paper-read%20online-2f5f9e)](https://thanatchon36.github.io/bp-from-scratch/paper/)
[![PDF](https://img.shields.io/badge/paper-PDF-b2182b)](https://github.com/thanatchon36/bp-from-scratch/raw/main/paper/paper.pdf)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

📄 **[Read the paper on the web](https://thanatchon36.github.io/bp-from-scratch/paper/)** ·
⬇️ **[Download the PDF](https://github.com/thanatchon36/bp-from-scratch/raw/main/paper/paper.pdf)** ·
📝 **[Markdown mirror](paper/paper.md)** ·
🔁 **[Reproduce](#reproduce)**

**A from-scratch NumPy state-vector simulator and a reproducible numerical study of barren plateaus.** Barren
plateaus are the exponential vanishing of gradients that makes deep, randomly initialised variational quantum
circuits untrainable. The literature makes precise, testable predictions about them, and checking those
predictions needs tens of thousands of cheap gradient evaluations on 2–16 qubits, nothing more. This repository
does that with a transparent ~500-line simulator, and the accompanying report measures where theory and numerics
agree and where they part.

- **`qsim`:** a state-vector simulator with in-place gate application, parameter-shift and adjoint gradients
  (the full gradient in `O(#gates)` state-vector operations) and exact dynamical Lie algebras. It is tested
  against independent references and PennyLane.
- **Experiments E0–E7:** Haar references, depth and cost dependence, initialisation, training under shot noise,
  QAOA and Lie-algebraic predictions. Every data point in the paper is cached in [`results/`](results/).
- **The paper:** *Barren Plateaus from Scratch* (14 pages), with every figure and table regenerated from the
  cached results by one command.

![E1: gradient variance versus qubit number, depth and kurtosis](results/figures/fig_e1.png)

## Install and quickstart

```bash
pip install -e ".[dev]"            # Python >= 3.10: numpy, scipy, pytest, matplotlib
pip install -e ".[dev,compare]"    # + PennyLane, enables the cross-check tests
```

```python
import numpy as np
from qsim import Circuit, PauliSum, adjoint_gradient

n, layers = 6, 4
c = Circuit(n)
for _ in range(layers):
    for q in range(n):
        c.ry(q)                       # every call creates a fresh parameter
    for q in range(n - 1):
        c.cnot(q, q + 1)

H = PauliSum.global_z(n)             # Z (x) Z (x) ... (x) Z
rng = np.random.default_rng(0)
theta = c.random_params(rng)

psi = c.run(theta)                    # Statevector
cost = H.expectation(psi)
grad = adjoint_gradient(c, theta, H)  # d cost / d theta, all parameters at once
print(cost, np.var(grad))
```

QAOA for MaxCut with shared parameters:

```python
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
H = PauliSum.maxcut(4, edges)         # cut size, to be maximised
c = Circuit(4)
for q in range(4):
    c.h(q)
for p in range(2):                    # p = 2 layers, parameters (gamma_0, beta_0, gamma_1, beta_1)
    for i, j in edges:
        c.rzz(i, j, param=2 * p, scale=2.0)      # exp(-i gamma Z_i Z_j)
    for q in range(4):
        c.rx(q, param=2 * p + 1, scale=2.0)      # exp(-i beta X_q)
```

| Command | Purpose |
|---|---|
| `make test` | 58 tests (2 of them need PennyLane) |
| `make bench` | Gate and gradient benchmarks against PennyLane |
| `make results` | Re-run every experiment (~1 CPU hour; resumable, cached in `results/`) |
| `make figures` | Figures and LaTeX tables from the cached results |
| `make paper` | `paper/paper.pdf`, the web reader `docs/paper/` and `paper/paper.md` |
| `make arxiv` | The arXiv source tarball, test-compiled with pdfLaTeX |
| `make docs` | The MkDocs site served by GitHub Pages |

### Features of `qsim`

- **State vector** of `n` qubits as a flat `complex128` array. Gates act in place by reshaping to a
  `(2^q, 2, rest)` or `(2^lo, 2, mid, 2, rest)` tensor, so no `2^n x 2^n` matrix is ever built.
- **Gates:** arbitrary 1-, 2- and k-qubit unitaries, with fast paths for diagonal gates, CNOT and CZ. Pauli
  strings, and rotations `exp(-i a G / 2)` for any involutory generator (RX, RY, RZ, RXX, RYY, RZZ, Pauli strings).
- **Parameterised circuits** with shared and scaled parameters, so QAOA needs no special case.
- **Cost operators:** Pauli sums (MaxCut, local and global Z) and the `|0...0><0...0|` projector.
- **Gradients:** parameter-shift rule, adjoint differentiation, and finite differences for testing.
- **Entanglement:** reduced density matrices and von Neumann entropy of any subset of qubits.
- **Haar-random unitaries and states** for 2-design reference values.

### Conventions

- Qubit 0 is the **most significant** bit: `psi[j]` with `j = sum_i b_i 2^(n-1-i)`. This is PennyLane's
  convention; Qiskit uses the reverse.
- Rotations are `exp(-i * angle * G / 2)`, as in PennyLane and Qiskit, so the parameter-shift rule with shift
  `pi/2` is exact.
- Two-qubit matrices are written in the `|q_first q_second>` basis, and `CNOT` has the first qubit as control.

## Research: *Barren Plateaus from Scratch* (2026)

The paper is in [`paper/`](paper/): a PDF in [`paper/paper.pdf`](paper/paper.pdf), a
[web version](https://thanatchon36.github.io/bp-from-scratch/paper/) (built into [`docs/paper/`](docs/paper/))
and a Markdown mirror in [`paper/paper.md`](paper/paper.md). The LaTeX source is arXiv-ready; see
[`paper/ARXIV.md`](paper/ARXIV.md). Main findings:

- **Haar references:** closed-form 2-design gradient variances for a random circuit before, after or on both
  sides of a rotation, verified numerically (Appendix A).
- **Approach to the plateau:** a hardware-efficient ansatz approaches the two-sided Haar value
  **non-monotonically** with depth. Its gradient distribution is **heavy-tailed** (kurtosis up to ~80) until it
  does, and the variance saturates while the half-chain entanglement is still below 40% of the Page value.
- **Cost and initialisation:** global costs decay as `4^-n`, while local costs decay with exponents −0.3 to −0.5,
  as Cerezo et al. predict. Small-angle Gaussian initialisation gives 10x larger gradients at n = 14, and identity
  blocks give a deterministic O(1/n) gradient.
- **Optimisation:** with exact gradients Adam trains the global cost even at n = 14, where |∇C| ≈ 1e-4. With
  10^4-shot noise it fails beyond n = 10, while the local cost keeps succeeding.
- **QAOA** for MaxCut shows no plateau at fixed depth p.
- **Lie algebras:** `{Y_i, Z_iZ_{i+1}}` closes exactly into **so(2n)**, of dimension n(2n−1), which makes it a
  free-fermion circuit. The TFIM HVA closes into an n²-dimensional algebra and `{X_i, Y_i, Z_iZ_{i+1}}` into
  su(2^n). The formula Var[C] = P(ρ)P(H)/dim g of Ragone et al. holds within ~10% where the circuit is a 2-design
  on e^g.

### Reproduce

```bash
make results   # BUDGET=100000 bash experiments/run_all.sh: all experiments, cached in results/
make paper     # figures and tables from results/, then the PDF (Tectonic), web reader and Markdown mirror (pandoc)
make arxiv     # dist/arxiv/bp-from-scratch-arxiv.tar.gz
```

Experiment scripts are in `experiments/` (`e0_haar_check.py` to `e7_dla.py`), ansätze in `ansatz/`, and the DLA
code in `qsim/dla.py`. Every experiment script skips cached points, so an interrupted run resumes where it stopped.

## Tests and benchmarks

```bash
python -m pytest -q
python benchmarks/bench_gates.py --max-n 18 --layers 10
```

The 58 tests check gate application against an independent `tensordot` reference, Kronecker-product matrices and
PennyLane. They also check the three gradient methods against each other, including shared and scaled
parameters, Hamiltonians against dense matrices, and entropies against Bell, GHZ and Page values.

![benchmark](benchmarks/bench.png)

The benchmark ran on a single core with NumPy 2.5 and Python 3.12 (see `benchmarks/machine.txt`). For the
hardware-efficient ansatz with 10 layers, the pure-NumPy forward pass is 2–8x faster than PennyLane's
`default.qubit`. The compiled `lightning.qubit` wins beyond 14 qubits. The adjoint method returns the full
gradient of 320 parameters at `n = 16` in ~0.7 s, which is what makes the variance experiments cheap.

## Repository layout

```
qsim/          state-vector simulator: state, gates, circuit, gradient (parameter-shift, adjoint), hamiltonian, dla
ansatz/        McClean HEA, RY-RZ HEA / ALA, identity blocks, Pauli-rotation HEA, QAOA MaxCut, HVA TFIM
experiments/   e0_haar_check ... e7_dla, common.py (caching, bootstrap, Adam, Haar formulas), run_all.sh
analysis/      make_figures.py, make_tables.py
results/       cached JSON for every data point in the paper + PNG figures
benchmarks/    bench_gates.py, results.csv, bench.png
paper/         main.tex, paper.pdf, paper.md, figures/, tables/, build.sh, web.py, arxiv.sh, ARXIV.md
docs/          MkDocs home page and the generated web reader of the paper (docs/paper/)
tests/         58 pytest tests (independent references, PennyLane cross-check, DLA algebra)
```

## Key references

- McClean et al., *Barren plateaus in quantum neural network training landscapes*, Nat. Commun. 9, 4812 (2018)
- Cerezo et al., *Cost function dependent barren plateaus in shallow parametrized quantum circuits*, Nat. Commun. 12, 1791 (2021)
- Grant et al., *An initialization strategy for addressing barren plateaus in parametrized quantum circuits*, Quantum 3, 214 (2019)
- Ortiz Marrero, Kieferová, Wiebe, *Entanglement-induced barren plateaus*, PRX Quantum 2, 040316 (2021)
- Larocca et al., *Diagnosing barren plateaus with tools from quantum optimal control*, Quantum 6, 824 (2022)
- Ragone et al., *A Lie algebraic theory of barren plateaus for deep parameterized quantum circuits*, Nat. Commun. 15, 7172 (2024)
- Jones & Gacon, *Efficient calculation of gradients in classical simulations of variational quantum algorithms*, arXiv:2009.02823

The paper's bibliography lists all 13 references with DOIs or arXiv identifiers.

## Citation

See [`CITATION.cff`](CITATION.cff). License: MIT.

```bibtex
@techreport{chongmankhong2026barren,
  author = {Chongmankhong, Thanatchon},
  title  = {Barren Plateaus from Scratch: A NumPy State-Vector Study of Trainability in Variational Quantum Circuits},
  year   = {2026},
  type   = {Technical report},
  url    = {https://thanatchon36.github.io/bp-from-scratch/paper/}
}
```

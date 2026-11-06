# bp-from-scratch

A from-scratch NumPy state-vector simulator (`qsim`) and a reproducible numerical study of **barren plateaus**:
the exponential vanishing of gradients that makes deep, randomly initialised variational quantum circuits
untrainable.

📄 **[Read the paper](paper/index.html)**: *Barren Plateaus from Scratch: A NumPy State-Vector Study of
Trainability in Variational Quantum Circuits*, Thanatchon Chongmankhong (2026) ([PDF](paper/paper.pdf)).

What the repository contains:

- **`qsim`**, a ~500-line simulator with in-place gate application and parameter-shift and adjoint gradients.
  It is validated against independent references and PennyLane.
- **Experiments E0–E7** on 2–16 qubits: Haar references, depth and cost dependence, initialisation, training
  under shot noise, QAOA and exact dynamical Lie algebras.
- **Cached results** for every data point in the paper, and scripts that regenerate every figure and table.

Source code, installation and reproduction steps are on
[GitHub](https://github.com/thanatchon36/bp-from-scratch).

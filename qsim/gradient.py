"""Gradients of the cost C(theta) = <0| U(theta)^dag H U(theta) |0>.

Three methods, all returning ``dC/dtheta`` as an array of length ``circuit.n_params``:

* :func:`parameter_shift_gradient` — exact, 2 circuit evaluations per *gate occurrence*
  (what quantum hardware would do).  Valid because every rotation has an involutory
  generator, so C is a sinusoid in each gate angle: dC/da = [C(a+pi/2) - C(a-pi/2)] / 2.
* :func:`adjoint_gradient` — exact, O(#gates) state-vector operations for the *whole*
  gradient (Jones & Gacon, arXiv:2009.02823).  This is the workhorse for experiments.
* :func:`finite_difference_gradient` — central differences, used only in tests.

Shared parameters and ``scale`` factors are handled by all three.
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit, Op
from .state import Statevector


def _apply_generator(state: Statevector, op: Op) -> Statevector:
    """Return G|psi> for the rotation generator of ``op`` (new state)."""
    out = state.copy()
    if op.diag:
        out.apply_diag(np.diagonal(op.generator), op.qubits)
    else:
        out.apply_matrix(op.generator, op.qubits)
    return out


def parameter_shift_gradient(circuit: Circuit, params, ham) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    grad = np.zeros(circuit.n_params)
    for k, op in enumerate(circuit.ops):
        if op.param is None:
            continue
        cp = circuit.expectation(params, ham, shifts={k: +np.pi / 2})
        cm = circuit.expectation(params, ham, shifts={k: -np.pi / 2})
        grad[op.param] += op.scale * 0.5 * (cp - cm)
    return grad


def adjoint_gradient(circuit: Circuit, params, ham, return_cost: bool = False):
    """Reverse-mode gradient.

    With psi_k = U_k ... U_1 |0> and lambda_k = U_{k+1}^dag ... U_M^dag H psi_M,
    dC/dtheta_k = 2 Re <lambda_k | dU_k/dtheta_k psi_{k-1}>
               = 2 Re [ (-i/2) * scale * <lambda_k | G_k psi_k> ].
    """
    params = np.asarray(params, dtype=float)
    psi = circuit.run(params)
    cost = ham.expectation(psi)
    lam = ham.apply(psi)
    grad = np.zeros(circuit.n_params)
    for k in range(len(circuit.ops) - 1, -1, -1):
        op = circuit.ops[k]
        if op.param is not None:
            mu = _apply_generator(psi, op)
            grad[op.param] += 2.0 * np.real((-0.5j) * op.scale * lam.inner(mu))
        Circuit.apply_op(psi, op, params, dagger=True)
        Circuit.apply_op(lam, op, params, dagger=True)
    return (cost, grad) if return_cost else grad


def finite_difference_gradient(circuit: Circuit, params, ham, h: float = 1e-5) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    grad = np.zeros(circuit.n_params)
    for k in range(circuit.n_params):
        p = params.copy()
        p[k] += h
        cp = circuit.expectation(p, ham)
        p[k] -= 2 * h
        cm = circuit.expectation(p, ham)
        grad[k] = (cp - cm) / (2 * h)
    return grad


def cost_and_grad(circuit: Circuit, params, ham):
    """Convenience wrapper for optimisers: returns (cost, gradient) via the adjoint method."""
    return adjoint_gradient(circuit, params, ham, return_cost=True)

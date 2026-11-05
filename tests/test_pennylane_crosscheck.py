"""Cross-check states, expectation values and gradients against PennyLane.

PennyLane uses the same conventions as qsim (wire 0 = most significant bit,
RX/RY/RZ/IsingZZ = exp(-i theta G / 2)), so results must agree to machine precision.
"""
import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
from pennylane import numpy as pnp  # noqa: E402

from qsim import Circuit, PauliSum, adjoint_gradient, parameter_shift_gradient  # noqa: E402

N = 4
N_PARAMS = 3 * N + 1


def qsim_circuit():
    c = Circuit(N)
    k = 0
    for q in range(N):
        c.h(q)
    for q in range(N):
        c.ry(q, param=k)
        k += 1
    for q in range(N - 1):
        c.cnot(q, q + 1)
    for q in range(N):
        c.rz(q, param=k)
        k += 1
    c.rzz(0, 2, param=k)
    k += 1
    c.cz(1, 3)
    for q in range(N):
        c.rx(q, param=k)
        k += 1
    assert c.n_params == N_PARAMS
    return c


def pennylane_ops(params):
    k = 0
    for q in range(N):
        qml.Hadamard(wires=q)
    for q in range(N):
        qml.RY(params[k], wires=q)
        k += 1
    for q in range(N - 1):
        qml.CNOT(wires=[q, q + 1])
    for q in range(N):
        qml.RZ(params[k], wires=q)
        k += 1
    qml.IsingZZ(params[k], wires=[0, 2])
    k += 1
    qml.CZ(wires=[1, 3])
    for q in range(N):
        qml.RX(params[k], wires=q)
        k += 1


def test_state_matches_pennylane(rng):
    params = rng.uniform(0, 2 * np.pi, N_PARAMS)
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def f(p):
        pennylane_ops(p)
        return qml.state()

    assert np.allclose(np.asarray(f(params)), qsim_circuit().run(params).data)


def test_expectation_and_gradient_match_pennylane(rng):
    params = rng.uniform(0, 2 * np.pi, N_PARAMS)
    H = PauliSum([(1.0, "ZZII"), (0.5, "IXIY"), (-0.25, "ZIII")])
    obs = (1.0 * (qml.PauliZ(0) @ qml.PauliZ(1)) + 0.5 * (qml.PauliX(1) @ qml.PauliY(3)) - 0.25 * qml.PauliZ(0))
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, diff_method="parameter-shift")
    def f(p):
        pennylane_ops(p)
        return qml.expval(obs)

    c = qsim_circuit()
    p = pnp.array(params, requires_grad=True)
    assert np.isclose(float(f(p)), c.expectation(params, H))
    g_pl = np.asarray(qml.grad(f)(p))
    assert np.allclose(g_pl, parameter_shift_gradient(c, params, H), atol=1e-8)
    assert np.allclose(g_pl, adjoint_gradient(c, params, H), atol=1e-8)

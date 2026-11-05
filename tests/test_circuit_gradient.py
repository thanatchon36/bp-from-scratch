import numpy as np
import pytest

from qsim import (Circuit, PauliSum, Projector0, Statevector, adjoint_gradient, cost_and_grad,
                  finite_difference_gradient, gates, parameter_shift_gradient)
from qsim.gates import random_unitary


# ----------------------------------------------------------------- Hamiltonians
def test_paulisum_expectation_and_apply_match_dense(rng):
    H = PauliSum([(0.5, "ZZII"), (-1.2, "IXYI"), (0.7, "ZIZZ"), (0.3, "IIII")])
    s = Statevector.random(4, rng)
    M = H.to_matrix()
    assert np.isclose(H.expectation(s), np.real(np.vdot(s.data, M @ s.data)))
    assert np.allclose(H.apply(s).data, M @ s.data)
    assert np.isclose(H.frobenius_norm_sq(), np.trace(M @ M).real)


def test_paulisum_algebra():
    a = PauliSum([(1.0, "ZI"), (2.0, "IZ")])
    b = PauliSum([(-1.0, "ZI"), (0.5, "XX")])
    s = (a + b).simplify()
    labels = dict((l, c) for c, l in s.terms)
    assert labels == {"IZ": 2.0, "XX": 0.5}
    assert np.allclose((2 * a).to_matrix(), 2 * a.to_matrix())


def test_maxcut_counts_cut_edges():
    n, edges = 3, [(0, 1), (1, 2), (0, 2)]
    H = PauliSum.maxcut(n, edges)
    for bits, cut in [("000", 0), ("001", 2), ("010", 2), ("111", 0), ("101", 2)]:
        e = np.zeros(2**n)
        e[int(bits, 2)] = 1
        assert np.isclose(H.expectation(Statevector(n, e)), cut), bits


def test_local_and_global_z_on_zero_state():
    s = Statevector.zero(5)
    assert np.isclose(PauliSum.global_z(5).expectation(s), 1.0)
    assert np.isclose(PauliSum.local_z(5).expectation(s), 1.0)
    assert np.isclose(PauliSum.local_z(5, normalize=False).expectation(s), 5.0)


def test_projector0(rng):
    s = Statevector.random(3, rng)
    P = Projector0(3)
    assert np.isclose(P.expectation(s), abs(s.data[0]) ** 2)
    assert np.allclose(P.apply(s).data, P.to_matrix() @ s.data)


# ----------------------------------------------------------------- circuits
def test_circuit_matches_manual_statevector(rng):
    c = Circuit(3).h(0).cnot(0, 1).ry(2, angle=0.3).rzz(1, 2, angle=0.8).cz(0, 2)
    assert c.n_params == 0
    s = Statevector.zero(3).apply_1q(gates.H, 0).apply_cnot(0, 1).apply_1q(gates.ry(0.3), 2)
    s.apply_2q(gates.rzz(0.8), 1, 2).apply_cz(0, 2)
    assert np.allclose(c.run().data, s.data)


def test_circuit_to_matrix_is_unitary(rng):
    c = Circuit(3).h(0).ry(1).cnot(0, 2).rx(2).rzz(0, 1)
    U = c.to_matrix(c.random_params(rng))
    assert np.allclose(U @ U.conj().T, np.eye(8))


def test_parameter_bookkeeping():
    c = Circuit(2)
    c.ry(0)  # p0
    c.ry(1, param=0)  # shared
    c.rz(0, param=3)  # jumps to 4 params
    c.rx(1)  # p4
    assert c.n_params == 5
    assert [op.param for op in c.ops] == [0, 0, 3, 4]
    with pytest.raises(ValueError):
        c.ry(0, param=1, angle=0.2)
    with pytest.raises(ValueError):
        c.rotation(gates.H + gates.X, (0,))  # not involutory


def test_param_dagger_round_trip(rng):
    c = Circuit(3).ry(0).rzz(0, 1, scale=2.0).rx(2, param=0).cnot(1, 2).unitary(random_unitary(4, rng), (0, 2))
    p = c.random_params(rng)
    s = c.run(p)
    for op in reversed(c.ops):
        Circuit.apply_op(s, op, p, dagger=True)
    assert np.allclose(s.data, Statevector.zero(3).data)


# ----------------------------------------------------------------- gradients
def build_mixed_circuit(n, rng):
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for q in range(n):
        c.ry(q)  # fresh parameters 0..n-1
    c.cnot(0, 1).cz(2, 3).cnot(3, 1)
    for q in range(n):
        c.rx(q, param=0, scale=0.5)  # shared + scaled
    c.rzz(0, 2).rzz(1, 3, param=1, scale=2.0)
    for q in range(n):
        c.rz(q)
    c.unitary(random_unitary(4, rng), (1, 2))
    for q in range(n):
        c.ry(q, angle=0.3)  # fixed angle, no parameter
    c.rxx(2, 3).pauli_rotation("XY", (0, 3))
    return c


@pytest.mark.parametrize("ham_name", ["pauli", "projector", "maxcut"])
def test_gradient_methods_agree(rng, ham_name):
    n = 4
    c = build_mixed_circuit(n, rng)
    params = c.random_params(rng)
    ham = {
        "pauli": PauliSum([(1.0, "ZZII"), (0.5, "IXXI"), (-0.3, "YIIZ"), (0.2, "IIZI")]),
        "projector": Projector0(n),
        "maxcut": PauliSum.maxcut(n, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
    }[ham_name]
    g_ps = parameter_shift_gradient(c, params, ham)
    g_ad = adjoint_gradient(c, params, ham)
    g_fd = finite_difference_gradient(c, params, ham)
    assert g_ps.shape == (c.n_params,)
    assert np.linalg.norm(g_ps) > 1e-3  # non-trivial gradient
    assert np.allclose(g_ps, g_fd, atol=1e-6)
    assert np.allclose(g_ad, g_ps, atol=1e-10)


def test_qaoa_shared_parameters(rng):
    n, p = 4, 2
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    H = PauliSum.maxcut(n, edges)
    c = Circuit(n)
    for q in range(n):
        c.h(q)
    for layer in range(p):
        g, b = 2 * layer, 2 * layer + 1
        for (i, j) in edges:
            c.rzz(i, j, param=g, scale=2.0)  # exp(-i gamma Z_i Z_j)
        for q in range(n):
            c.rx(q, param=b, scale=2.0)  # exp(-i beta X_q)
    assert c.n_params == 2 * p
    params = c.random_params(rng)
    assert np.allclose(parameter_shift_gradient(c, params, H), finite_difference_gradient(c, params, H), atol=1e-6)
    assert np.allclose(adjoint_gradient(c, params, H), finite_difference_gradient(c, params, H), atol=1e-6)


def test_no_parameters_gives_empty_gradient():
    c = Circuit(2).h(0).cnot(0, 1).ry(1, angle=0.1)
    H = PauliSum.global_z(2)
    assert parameter_shift_gradient(c, np.array([]), H).shape == (0,)
    assert adjoint_gradient(c, np.array([]), H).shape == (0,)


def test_cost_and_grad(rng):
    c = Circuit(3).ry(0).ry(1).ry(2).cnot(0, 1).cnot(1, 2).rz(2)
    H = PauliSum.local_z(3)
    p = c.random_params(rng)
    cost, g = cost_and_grad(c, p, H)
    assert np.isclose(cost, c.expectation(p, H))
    assert np.allclose(g, finite_difference_gradient(c, p, H), atol=1e-6)

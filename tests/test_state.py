import numpy as np
import pytest

from qsim import Statevector, gates
from qsim.gates import pauli_matrix, random_unitary


# ----------------------------------------------------------------- independent references
def ref_apply_1q(psi, U, q, n):
    t = psi.reshape((2,) * n)
    r = np.tensordot(U, t, axes=([1], [q]))
    return np.moveaxis(r, 0, q).reshape(-1)


def ref_apply_2q(psi, U, q1, q2, n):
    t = psi.reshape((2,) * n)
    r = np.tensordot(U.reshape(2, 2, 2, 2), t, axes=([2, 3], [q1, q2]))
    return np.moveaxis(r, [0, 1], [q1, q2]).reshape(-1)


def ref_cnot_matrix(n, c, t):
    d = 2**n
    m = np.zeros((d, d))
    for j in range(d):
        bits = [(j >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[c]:
            bits[t] ^= 1
        i = sum(b << (n - 1 - k) for k, b in enumerate(bits))
        m[i, j] = 1
    return m


# ----------------------------------------------------------------- gate matrices
@pytest.mark.parametrize("fn", [gates.rx, gates.ry, gates.rz, gates.rxx, gates.ryy, gates.rzz])
def test_rotations_are_unitary(fn):
    U = fn(0.7321)
    assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]))


def test_rotation_at_pi_equals_minus_i_generator():
    assert np.allclose(gates.rx(np.pi), -1j * gates.X)
    assert np.allclose(gates.ry(np.pi), -1j * gates.Y)
    assert np.allclose(gates.rz(np.pi), -1j * gates.Z)
    assert np.allclose(gates.rzz(np.pi), -1j * gates.ZZ)


def test_pauli_matrix_kron_order():
    assert np.allclose(pauli_matrix("XZ"), np.kron(gates.X, gates.Z))
    assert np.allclose(pauli_matrix("IYI"), np.kron(np.kron(gates.I2, gates.Y), gates.I2))


def test_random_unitary_is_unitary(rng):
    U = random_unitary(8, rng)
    assert np.allclose(U @ U.conj().T, np.eye(8))


# ----------------------------------------------------------------- one- and two-qubit application
@pytest.mark.parametrize("q", range(5))
def test_apply_1q_matches_reference(rng, q):
    n = 5
    s = Statevector.random(n, rng)
    U = random_unitary(2, rng)
    expected = ref_apply_1q(s.data.copy(), U, q, n)
    s.apply_1q(U, q)
    assert np.allclose(s.data, expected)


@pytest.mark.parametrize("q1,q2", [(0, 1), (1, 0), (0, 4), (4, 0), (2, 3), (3, 1), (1, 3)])
def test_apply_2q_matches_reference(rng, q1, q2):
    n = 5
    s = Statevector.random(n, rng)
    U = random_unitary(4, rng)
    expected = ref_apply_2q(s.data.copy(), U, q1, q2, n)
    s.apply_2q(U, q1, q2)
    assert np.allclose(s.data, expected)


@pytest.mark.parametrize("q1,q2", [(0, 1), (1, 0), (0, 3), (3, 0), (2, 1)])
def test_fast_paths_match_general_path(rng, q1, q2):
    n = 4
    s0 = Statevector.random(n, rng)
    # CNOT
    a = s0.copy().apply_cnot(q1, q2)
    b = s0.copy().apply_2q(gates.CNOT, q1, q2)
    c = ref_cnot_matrix(n, q1, q2) @ s0.data
    assert np.allclose(a.data, b.data) and np.allclose(a.data, c)
    # CZ
    a = s0.copy().apply_cz(q1, q2)
    b = s0.copy().apply_2q(gates.CZ, q1, q2)
    assert np.allclose(a.data, b.data)
    # diagonal two-qubit rotation
    U = gates.rzz(1.234)
    a = s0.copy().apply_diag_2q(np.diagonal(U), q1, q2)
    b = s0.copy().apply_2q(U, q1, q2)
    assert np.allclose(a.data, b.data)


def test_diag_1q_matches_general(rng):
    s0 = Statevector.random(4, rng)
    U = gates.rz(0.4)
    assert np.allclose(s0.copy().apply_diag_1q(np.diagonal(U), 2).data, s0.copy().apply_1q(U, 2).data)


def test_kq_path_matches_2q_and_kron(rng):
    n = 5
    s0 = Statevector.random(n, rng)
    U2 = random_unitary(4, rng)
    assert np.allclose(s0.copy()._apply_kq(U2, (1, 3)).data, s0.copy().apply_2q(U2, 1, 3).data)
    assert np.allclose(s0.copy()._apply_kq(U2, (3, 1)).data, s0.copy().apply_2q(U2, 3, 1).data)
    U3 = random_unitary(8, rng)
    expected = np.kron(U3, np.eye(4)) @ s0.data  # qubits (0,1,2) are the leading axes
    assert np.allclose(s0.copy().apply_matrix(U3, (0, 1, 2)).data, expected)


def test_pauli_string_matches_matrix(rng):
    n = 4
    for label in ["XIZY", "YYYY", "IZIX", "ZXYI", "IIII"]:
        s0 = Statevector.random(n, rng)
        expected = pauli_matrix(label) @ s0.data
        assert np.allclose(s0.copy().apply_pauli_string(label).data, expected), label


# ----------------------------------------------------------------- known states
def test_bell_state():
    s = Statevector.zero(2).apply_1q(gates.H, 0).apply_cnot(0, 1)
    assert np.allclose(s.data, [1, 0, 0, 1] / np.sqrt(2))


def test_ghz_state():
    n = 6
    s = Statevector.zero(n).apply_1q(gates.H, 0)
    for q in range(n - 1):
        s.apply_cnot(q, q + 1)
    expected = np.zeros(2**n)
    expected[0] = expected[-1] = 1 / np.sqrt(2)
    assert np.allclose(s.data, expected)
    assert np.isclose(s.norm(), 1.0)


def test_amplitude_and_probabilities():
    s = Statevector.zero(3).apply_1q(gates.X, 2)  # |001>
    assert np.isclose(abs(s.amplitude("001")), 1.0)
    assert np.isclose(s.probabilities().sum(), 1.0)


# ----------------------------------------------------------------- entanglement entropy
def test_entropy_product_state():
    assert np.isclose(Statevector.zero(4).entanglement_entropy([0, 1]), 0.0)


def test_entropy_bell_and_ghz():
    bell = Statevector.zero(2).apply_1q(gates.H, 0).apply_cnot(0, 1)
    assert np.isclose(bell.entanglement_entropy([0]), np.log(2))
    assert np.isclose(bell.entanglement_entropy([1], base=2), 1.0)
    n = 6
    ghz = Statevector.zero(n).apply_1q(gates.H, 0)
    for q in range(n - 1):
        ghz.apply_cnot(q, q + 1)
    assert np.isclose(ghz.entanglement_entropy(range(n // 2)), np.log(2))


def test_entropy_haar_state_near_page_value(rng):
    """Page (1993): for m = n/2, <S> ~ m ln 2 - 1/2 (nats); fluctuations are tiny at n = 10."""
    n, m = 10, 5
    s = Statevector.random(n, rng)
    page = m * np.log(2) - 0.5
    assert abs(s.entanglement_entropy(range(m)) - page) < 0.05


def test_reduced_density_matrix(rng):
    s = Statevector.random(5, rng)
    rho = s.reduced_density_matrix([1, 3])
    assert rho.shape == (4, 4)
    assert np.isclose(np.trace(rho).real, 1.0)
    assert np.allclose(rho, rho.conj().T)

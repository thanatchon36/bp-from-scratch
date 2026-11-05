import numpy as np

from qsim.dla import (LieClosure, anticommute, commutator_i, label_to_pauli, operator_from_terms, pauli_mult,
                      pauli_string_closure, pauli_to_label, product_state_operator, set_purity)
from qsim.gates import pauli_matrix


def test_label_roundtrip():
    for lab in ["IXYZ", "ZZ", "YIX", "I"]:
        assert pauli_to_label(label_to_pauli(lab), len(lab)) == lab


def test_pauli_mult_matches_matrices(rng):
    labels = ["XZ", "YY", "IX", "ZY", "XI", "YZ"]
    for a in labels:
        for b in labels:
            k, c = pauli_mult(label_to_pauli(a), label_to_pauli(b))
            lhs = pauli_matrix(a) @ pauli_matrix(b)
            rhs = (1j**k) * pauli_matrix(pauli_to_label(c, 2))
            assert np.allclose(lhs, rhs), (a, b)
            assert anticommute(label_to_pauli(a), label_to_pauli(b)) == (not np.allclose(lhs, pauli_matrix(b) @ pauli_matrix(a)))


def test_commutator_i_matches_matrices():
    A = operator_from_terms([(0.7, "XZ"), (-1.1, "YI")])
    B = operator_from_terms([(0.4, "ZZ"), (2.0, "IX")])
    C = commutator_i(A, B)
    MA = sum(c * pauli_matrix("XZ" if p == label_to_pauli("XZ") else "YI") for p, c in A.items())
    MB = 0.4 * pauli_matrix("ZZ") + 2.0 * pauli_matrix("IX")
    MC = sum(c * pauli_matrix(pauli_to_label(p, 2)) for p, c in C.items())
    assert np.allclose(MC, 1j * (MA @ MB - MB @ MA))


def test_closure_dimensions():
    # {X, Z} on one qubit generates su(2)
    assert len(pauli_string_closure([label_to_pauli("X"), label_to_pauli("Z")])) == 3
    # {X1, Y1, X2, Y2, Z1Z2} generates su(4)
    gens = [label_to_pauli(s) for s in ["XI", "YI", "IX", "IY", "ZZ"]]
    assert len(pauli_string_closure(gens)) == 15
    # generic closure agrees with the string closure
    lc = LieClosure([operator_from_terms([(1.0, s)]) for s in ["XI", "YI", "IX", "IY", "ZZ"]])
    assert lc.dim == 15
    # commuting generators: dimension = number of generators
    lc = LieClosure([operator_from_terms([(1.0, "ZI")]), operator_from_terms([(1.0, "IZ")])])
    assert lc.dim == 2


def test_purities():
    n = 2
    strings = pauli_string_closure([label_to_pauli(s) for s in ["XI", "YI", "IX", "IY", "ZZ"]])  # su(4)
    rho = product_state_operator(n, "0")  # |00><00|
    # P_su(d)(rho) = Tr(rho^2) - 1/d = 1 - 1/4
    assert np.isclose(set_purity(strings, rho, n), 1 - 0.25)
    H = operator_from_terms([(1.0, "ZZ")])
    assert np.isclose(set_purity(strings, H, n), 4.0)  # ||ZZ||_F^2 = 2^n
    lc = LieClosure([operator_from_terms([(1.0, s)]) for s in ["XI", "YI", "IX", "IY", "ZZ"]])
    assert np.isclose(lc.purity(rho, n), 0.75) and np.isclose(lc.purity(H, n), 4.0)

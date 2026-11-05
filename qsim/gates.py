"""Gate matrices, rotation generators and Haar-random unitaries.

Conventions (identical to PennyLane's)
--------------------------------------
* Qubit 0 is the most significant (leftmost) qubit: |q0 q1 ... q_{n-1}>.
* A two-qubit matrix acting on qubits (a, b) is written in the basis |q_a q_b>,
  i.e. the *first* qubit of the gate is the more significant bit of the 4x4
  matrix index.  CNOT below has the first qubit as control.
* Rotation gates are R_G(theta) = exp(-i * theta * G / 2) for a Hermitian,
  involutory generator G (G^2 = I).  This is the convention of RX, RY, RZ,
  IsingZZ, ... in PennyLane and Qiskit, and it is what makes the
  parameter-shift rule with shift pi/2 exact.
"""
from __future__ import annotations

import numpy as np

C = np.complex128

I2 = np.eye(2, dtype=C)
X = np.array([[0, 1], [1, 0]], dtype=C)
Y = np.array([[0, -1j], [1j, 0]], dtype=C)
Z = np.array([[1, 0], [0, -1]], dtype=C)
H = np.array([[1, 1], [1, -1]], dtype=C) / np.sqrt(2.0)
S = np.array([[1, 0], [0, 1j]], dtype=C)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=C)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

# first qubit = control
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=C)
CZ = np.diag([1, 1, 1, -1]).astype(C)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=C)
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)


def pauli_matrix(label: str) -> np.ndarray:
    """Dense matrix of a Pauli string such as ``'XZI'`` (leftmost = qubit 0)."""
    m = np.array([[1.0 + 0j]], dtype=C)
    for ch in label.upper():
        m = np.kron(m, PAULI[ch])
    return m


def is_involutory(g: np.ndarray, tol: float = 1e-12) -> bool:
    """True if g @ g == I (required for the pi/2 parameter-shift rule)."""
    g = np.asarray(g)
    return np.allclose(g @ g, np.eye(g.shape[0]), atol=tol)


def is_diagonal(m: np.ndarray, tol: float = 1e-12) -> bool:
    m = np.asarray(m)
    return np.allclose(m, np.diag(np.diagonal(m)), atol=tol)


def rotation(generator: np.ndarray, angle: float) -> np.ndarray:
    """exp(-i * angle * G / 2) for an involutory Hermitian generator G."""
    generator = np.asarray(generator, dtype=C)
    d = generator.shape[0]
    return np.cos(angle / 2) * np.eye(d, dtype=C) - 1j * np.sin(angle / 2) * generator


def rx(a: float) -> np.ndarray:
    return rotation(X, a)


def ry(a: float) -> np.ndarray:
    return rotation(Y, a)


def rz(a: float) -> np.ndarray:
    return rotation(Z, a)


def rxx(a: float) -> np.ndarray:
    return rotation(XX, a)


def ryy(a: float) -> np.ndarray:
    return rotation(YY, a)


def rzz(a: float) -> np.ndarray:
    return rotation(ZZ, a)


def random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR decomposition of a Ginibre matrix (Mezzadri 2007)."""
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))

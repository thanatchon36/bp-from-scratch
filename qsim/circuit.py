"""Parameterized quantum circuits.

A :class:`Circuit` is an ordered list of :class:`Op`.  An op is either a fixed
gate (``matrix``) or a rotation ``exp(-i * angle * G / 2)`` with an involutory
generator ``G``.  A rotation's angle is either fixed (``angle``) or
``scale * params[param]``; several ops may share one parameter index (as in
QAOA, where every ZZ rotation of a layer uses the same gamma).

Builder methods return ``self`` so circuits can be written fluently::

    c = Circuit(3).h(0).cnot(0, 1).ry(2)            # ry(2) creates parameter 0
    c.rzz(0, 1, param=0, scale=2.0)                  # shares parameter 0
    psi = c.run(params)                              # Statevector
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from . import gates as G
from .gates import C, is_diagonal, is_involutory, rotation
from .state import Statevector


@dataclass
class Op:
    name: str
    qubits: tuple
    matrix: Optional[np.ndarray] = None  # fixed gate
    generator: Optional[np.ndarray] = None  # rotation generator (G^2 = I)
    angle: Optional[float] = None  # fixed rotation angle
    param: Optional[int] = None  # parameter index (rotation only)
    scale: float = 1.0  # angle = scale * params[param]
    diag: bool = False  # matrix / generator is diagonal -> fast path

    @property
    def is_rotation(self) -> bool:
        return self.generator is not None

    @property
    def is_parametrized(self) -> bool:
        return self.param is not None

    def angle_at(self, params=None, shift: float = 0.0) -> float:
        if self.generator is None:
            raise ValueError(f"{self.name} is not a rotation")
        if self.param is None:
            return float(self.angle) + shift
        if params is None:
            raise ValueError("circuit has parameters but none were given")
        return self.scale * float(params[self.param]) + shift

    def matrix_at(self, params=None, shift: float = 0.0) -> np.ndarray:
        if self.generator is None:
            return self.matrix
        return rotation(self.generator, self.angle_at(params, shift))

    def __repr__(self) -> str:
        if self.generator is None:
            return f"{self.name}{self.qubits}"
        if self.param is None:
            return f"{self.name}{self.qubits}[angle={self.angle:.4g}]"
        s = f"{self.scale:g}*" if self.scale != 1.0 else ""
        return f"{self.name}{self.qubits}[{s}p{self.param}]"


class Circuit:
    def __init__(self, n: int):
        self.n = int(n)
        self.ops: list[Op] = []
        self.n_params = 0

    # ------------------------------------------------------------------ bookkeeping
    def __len__(self) -> int:
        return len(self.ops)

    @property
    def n_gates(self) -> int:
        return len(self.ops)

    @property
    def parametrized_ops(self) -> list[Op]:
        return [op for op in self.ops if op.param is not None]

    def add_param(self) -> int:
        k = self.n_params
        self.n_params += 1
        return k

    def random_params(self, rng: np.random.Generator, low: float = 0.0, high: float = 2 * np.pi) -> np.ndarray:
        return rng.uniform(low, high, size=self.n_params)

    def _check_qubits(self, qubits) -> tuple:
        qubits = tuple(int(q) for q in qubits)
        for q in qubits:
            if not 0 <= q < self.n:
                raise ValueError(f"qubit {q} out of range for n={self.n}")
        if len(set(qubits)) != len(qubits):
            raise ValueError("repeated qubit in gate")
        return qubits

    # ------------------------------------------------------------------ fixed gates
    def gate(self, matrix, qubits, name: str = "U") -> "Circuit":
        qubits = self._check_qubits(qubits)
        matrix = np.asarray(matrix, dtype=C)
        d = 2 ** len(qubits)
        if matrix.shape != (d, d):
            raise ValueError(f"matrix shape {matrix.shape} does not fit {len(qubits)} qubit(s)")
        self.ops.append(Op(name, qubits, matrix=matrix, diag=is_diagonal(matrix)))
        return self

    unitary = gate

    def h(self, q: int) -> "Circuit":
        return self.gate(G.H, (q,), "H")

    def x(self, q: int) -> "Circuit":
        return self.gate(G.X, (q,), "X")

    def y(self, q: int) -> "Circuit":
        return self.gate(G.Y, (q,), "Y")

    def z(self, q: int) -> "Circuit":
        return self.gate(G.Z, (q,), "Z")

    def s(self, q: int) -> "Circuit":
        return self.gate(G.S, (q,), "S")

    def t(self, q: int) -> "Circuit":
        return self.gate(G.T, (q,), "T")

    def cnot(self, control: int, target: int) -> "Circuit":
        return self.gate(G.CNOT, (control, target), "CNOT")

    def cz(self, a: int, b: int) -> "Circuit":
        return self.gate(G.CZ, (a, b), "CZ")

    def swap(self, a: int, b: int) -> "Circuit":
        return self.gate(G.SWAP, (a, b), "SWAP")

    # ------------------------------------------------------------------ rotations
    def rotation(self, generator, qubits, param: Optional[int] = None, angle: Optional[float] = None,
                 scale: float = 1.0, name: str = "R") -> "Circuit":
        """Append exp(-i * a * G / 2).  ``a`` is ``angle`` if given, else ``scale * params[param]``;
        with neither ``angle`` nor ``param`` a fresh parameter is created."""
        qubits = self._check_qubits(qubits)
        generator = np.asarray(generator, dtype=C)
        d = 2 ** len(qubits)
        if generator.shape != (d, d):
            raise ValueError("generator shape does not fit the qubits")
        if not is_involutory(generator):
            raise ValueError("generator must satisfy G^2 = I (needed for the parameter-shift rule)")
        if angle is not None and param is not None:
            raise ValueError("give either a fixed angle or a parameter index, not both")
        if angle is None:
            if param is None:
                param = self.add_param()
            else:
                param = int(param)
                if param < 0:
                    raise ValueError("parameter index must be >= 0")
                self.n_params = max(self.n_params, param + 1)
        self.ops.append(Op(name, qubits, generator=generator, angle=None if angle is None else float(angle),
                           param=param, scale=float(scale), diag=is_diagonal(generator)))
        return self

    def rx(self, q: int, **kw) -> "Circuit":
        return self.rotation(G.X, (q,), name="RX", **kw)

    def ry(self, q: int, **kw) -> "Circuit":
        return self.rotation(G.Y, (q,), name="RY", **kw)

    def rz(self, q: int, **kw) -> "Circuit":
        return self.rotation(G.Z, (q,), name="RZ", **kw)

    def rxx(self, a: int, b: int, **kw) -> "Circuit":
        return self.rotation(G.XX, (a, b), name="RXX", **kw)

    def ryy(self, a: int, b: int, **kw) -> "Circuit":
        return self.rotation(G.YY, (a, b), name="RYY", **kw)

    def rzz(self, a: int, b: int, **kw) -> "Circuit":
        return self.rotation(G.ZZ, (a, b), name="RZZ", **kw)

    def pauli_rotation(self, label: str, qubits, **kw) -> "Circuit":
        """exp(-i a P / 2) for a Pauli string P on the given qubits (e.g. 'XY' on (0, 3))."""
        return self.rotation(G.pauli_matrix(label), qubits, name=f"R{label}", **kw)

    # ------------------------------------------------------------------ execution
    @staticmethod
    def apply_op(state: Statevector, op: Op, params=None, shift: float = 0.0, dagger: bool = False) -> None:
        """Apply ``op`` (or its adjoint) in place."""
        if op.generator is None:
            if op.name == "CNOT":
                state.apply_cnot(*op.qubits)
            elif op.name == "CZ":
                state.apply_cz(*op.qubits)
            elif op.diag:
                d = np.diagonal(op.matrix)
                state.apply_diag(d.conj() if dagger else d, op.qubits)
            else:
                U = op.matrix.conj().T if dagger else op.matrix
                state.apply_matrix(U, op.qubits)
            return
        a = op.angle_at(params, shift)
        if dagger:
            a = -a
        if op.diag:
            state.apply_diag(np.exp(-0.5j * a * np.diagonal(op.generator).real), op.qubits)
        else:
            state.apply_matrix(rotation(op.generator, a), op.qubits)

    def run(self, params=None, shifts: Optional[Dict[int, float]] = None,
            state: Optional[Statevector] = None) -> Statevector:
        """Return U(params)|0...0> (or U applied to a copy of ``state``).

        ``shifts`` maps *op index* -> extra angle, used by the parameter-shift rule
        to shift a single occurrence of a (possibly shared) parameter."""
        psi = Statevector.zero(self.n) if state is None else state.copy()
        if params is not None:
            params = np.asarray(params, dtype=float)
            if params.shape != (self.n_params,):
                raise ValueError(f"expected {self.n_params} parameters, got shape {params.shape}")
        for k, op in enumerate(self.ops):
            shift = shifts.get(k, 0.0) if shifts else 0.0
            self.apply_op(psi, op, params, shift)
        return psi

    def expectation(self, params, ham, shifts: Optional[Dict[int, float]] = None) -> float:
        return ham.expectation(self.run(params, shifts))

    def to_matrix(self, params=None) -> np.ndarray:
        """Dense 2^n x 2^n unitary (small n only)."""
        d = 2**self.n
        U = np.zeros((d, d), dtype=C)
        for j in range(d):
            e = np.zeros(d, dtype=C)
            e[j] = 1.0
            U[:, j] = self.run(params, state=Statevector(self.n, e)).data
        return U

    def __repr__(self) -> str:
        return f"Circuit(n={self.n}, gates={len(self.ops)}, params={self.n_params})"

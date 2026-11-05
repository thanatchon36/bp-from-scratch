"""qsim — a small, from-scratch NumPy state-vector simulator for variational circuits."""
from . import gates
from .circuit import Circuit, Op
from .gradient import adjoint_gradient, cost_and_grad, finite_difference_gradient, parameter_shift_gradient
from .hamiltonian import PauliSum, Projector0
from .state import Statevector

__version__ = "0.1.0"
__all__ = [
    "gates",
    "Circuit",
    "Op",
    "Statevector",
    "PauliSum",
    "Projector0",
    "adjoint_gradient",
    "parameter_shift_gradient",
    "finite_difference_gradient",
    "cost_and_grad",
]

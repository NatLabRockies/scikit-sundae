"""
Bindings for the CVODE solver in SUNDIALS, used for solving systems of ordinary
differential equations (ODE). Features variable-order, variable-step integration
with support for direct and iterative linear solvers, root finding, and more.

"""

from ._precond import CVODEPrecond
from ._jactimes import CVODEJacTimes
from ._solver import CVODE, CVODEResult

__all__ = [
    'CVODE',
    'CVODEResult',
    'CVODEPrecond',
    'CVODEJacTimes',
]

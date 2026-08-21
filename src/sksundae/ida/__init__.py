"""
Bindings for the IDA solver in SUNDIALS, used for solving differential-algebraic
equation (DAE) systems. Supports adaptive time-stepping with support for direct
and iterative linear solvers, root finding, and more.

"""

from ._precond import IDAPrecond
from ._jactimes import IDAJacTimes
from ._solver import IDA, IDAResult

__all__ = [
    'IDA',
    'IDAResult',
    'IDAPrecond',
    'IDAJacTimes',
]

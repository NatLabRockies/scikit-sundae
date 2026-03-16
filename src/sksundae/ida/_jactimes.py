# ida._jactimes.py

from __future__ import annotations
from typing import Callable


class IDAJacTimes:
    """Jacobian-vector product."""

    __slots__ = ('setupfn', 'solvefn')

    def __init__(self, setupfn: Callable | None, solvefn: Callable) -> None:
        """
        Wrapper for passing Jacobian-vector product functions to IDA. The
        Jacobian-vector product interface is only supported by iterative solvers
        (gmres, bicgstab, tfqmr).

        Parameters
        ----------
        setupfn : Callable or None
            A function to setup data before solving the Jacobian-vector product.
            Use None if not needed. The required signature is in the notes.
        solvefn : Callable
            A function that solves for the Jacobian-vector product `J*v` (or
            an approximation to it). The required signature is in the notes.

        Raises
        ------
        TypeError
            'setupfn' must be type Callable or None.
        TypeError
            'solvefn' must be type Callable.

        Notes
        -----
        The solve and setup functions require specific function signatures. For
        'solvefn' use `f(t, y, yp, res, v, Jv, cj[, userdata])`. Any return
        values are ignored. Instead, the function should fill the pre-allocated
        memory for 'Jv' (a 1D array) with the solution (or approximation) to
        `J*v`. Use `[:]` to fill the array rather than overwriting it. For
        example, `Jv[:] = f(...)` is correct whereas `Jv = f(...)` is not.
        The user is responsible for managing their own Jacobian data if needed.
        'setupfn' can be used to help setup any needed values/data so that
        'solvefn' is not overly complex.

        'setupfn' requires the signature `f(t, y, yp, res, cj[, userdata])`.
        As with 'solvefn', any return values are ignored. However, you can use
        the function to define global variables or add them to `userdata` so
        they can be passed to 'solvefn'. The order of function calls is always
        'resfn' -> 'setupfn' -> 'solvefn' for each integration step.

        """

        if setupfn is None:
            pass
        elif not isinstance(setupfn, Callable):
            raise TypeError("'setupfn' must be type Callable.")

        if not isinstance(solvefn, Callable):
            raise TypeError("'solvefn' must be type Callable.")

        self.setupfn = setupfn
        self.solvefn = solvefn

# ida._precond.py

from __future__ import annotations
from typing import Callable


class IDAPrecond:
    """Preconditioner wrapper."""

    __slots__ = ('setupfn', 'solvefn', 'side', '_prectype')

    def __init__(self, setupfn: Callable | None, solvefn: Callable) -> None:
        """
        Wrapper for passing preconditioner functions to IDA. Preconditioning is
        only supported by iterative solvers (gmres, bicgstab, tfqmr). IDA only
        supports left preconditioning. Keep this in mind when defining your
        setup and solve functions.

        Parameters
        ----------
        setupfn : Callable or None, optional
            A function to setup data before solving the preconditioned problem.
            Use None if not needed. The required signature is in the notes.
        solvefn : Callable
            A function that solves the preconditioned problem `P*zvec = rvec`.
            P is a preconditioner matrix approximating the Jacobian, at least
            crudely. The required signature is in the notes.

        Raises
        ------
        TypeError
            'setupfn' must be type Callable or None.
        TypeError
            'solvefn' must be type Callable.

        Notes
        -----
        The solve and setup functions require specific function signatures. For
        'solvefn' use `f(t, y, yp, res, rvec, zvec, cj, delta[, userdata])`.
        Any return values are ignored. Instead, the function should fill the
        pre-allocated memory for 'zvec' with the solution to the preconditioned
        problem `P*zvec = rvec`. Don't forget to use `[:]` to fill the array
        rather than overwriting it. For example, `zvec[:] = f(...)` is correct
        whereas `zvec = f(...)` is not.

        Defining a preconditioning matrix is non-trivial and left to the user.
        For IDA, P should at least crudely approximate the Jacobian given by
        `J = dF_i/dy_j + cj*dF_i/dyp_j` where `res = F(t, y, yp)` is the
        residual function that describes the system of DAEs. If you need extra
        parameters or values, they can be passed via the optional `userdata`
        argument. For convenience, you can also define the optional 'setupfn'
        to setup any values (e.g., P) before the solve step.

        The 'setupfn' is an optional function that you can use to perform any
        operations needed before solving. The required signature for 'setupfn'
        is `f(t, y, yp, res, cj[, userdata])`. Any return values are ignored.
        However, you can use the function to define either global variables or
        add them to `userdata` so they can be passed to 'solvefn'. Please
        refer to the `SUNDIALS documentation`_ for more information about these
        functions and their input arguments.

        .. _SUNDIALS documentation: https://sundials.readthedocs.io/en/v6.1.1/ \
            ida/Usage/index.html#preconditioner-setup-iterative-linear-solvers

        """

        if setupfn is None:
            pass
        elif not isinstance(setupfn, Callable):
            raise TypeError("'setupfn' must be type Callable.")

        if not isinstance(solvefn, Callable):
            raise TypeError("'solvefn' must be type Callable.")

        self.setupfn = setupfn
        self.solvefn = solvefn
        self.side = 'left'  # IDA only supports left preconditioning

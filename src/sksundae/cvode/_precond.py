# cvode._precond.py

from __future__ import annotations
from typing import Callable


class CVODEPrecond:
    """Preconditioner wrapper."""

    __slots__ = ('setupfn', 'solvefn', 'side', '_prectype')

    def __init__(self, setupfn: Callable | None, solvefn: Callable,
                 side: str = 'left') -> None:
        """
        Wrapper for passing preconditioner functions to CVODE. Preconditioning
        is only supported by iterative solvers (gmres, bicgstab, tfqmr).

        Parameters
        ----------
        setupfn : Callable or None
            A function to setup data before solving the preconditioned problem.
            Use None if not needed. The required signature is in the notes.
        solvefn : Callable
            A function that solves the preconditioned problem `P*zvec = rvec`.
            P is a preconditioner matrix approximating `I - gamma*J`, at least
            crudely. The required signature is in the notes.
        side : {'left', 'right', 'both'}, optional
            The preconditioning type to use. Can be 'left', 'right', or 'both'.
            The default is 'left'.

        Raises
        ------
        TypeError
            'setupfn' must be type Callable or None.
        TypeError
            'solvefn' must be type Callable.
        ValueError
            'side' must be in {'left', 'right', 'both'}.

        Notes
        -----
        The solve and setup functions require specific function signatures. For
        'solvefn' use `f(t, y, yp, rvec, zvec, gamma, delta, lr[, userdata])`.
        Any return values are ignored. Instead, the function should fill the
        pre-allocated memory for 'zvec' with the solution to the preconditioned
        problem `P*zvec = rvec`. Don't forget to use `[:]` to fill the array
        rather than overwriting it. For example, `zvec[:] = f(...)` is correct
        whereas `zvec = f(...)` is not. The input `lr` is a flag specifying
        whether left (1) or right (2) preconditioning is being used. In cases
        where 'both' is selected for the preconditioning type, 'solvefn' will
        be called twice and you can control the behavior according to this flag.

        Defining a preconditioning matrix is non-trivial and left to the user.
        For CVODE, P should at least crudely approximate `I - gamma*J` where
        `I` is the identity and `J = df_i/dy_j` is the Jacobian defined by
        the system of ODEs `yp = f(t, y)`. If you need extra parameters or
        values, they can be passed via the optional `userdata` argument. For
        convenience, you can also define the optional 'setupfn' to setup any
        values (e.g., P) before the solve step.

        The 'setupfn' is an optional function that you can use to perform any
        operations needed before solving. The required signature for 'setupfn'
        is `f(t, y, yp, jok, jnew, gamma[, userdata])`. Any return values are
        ignored. However, you can use the function to define global variables or
        add them to `userdata` so they can be passed to 'solvefn'. The inputs
        `jok` and `jnew` are designed to help optimize performance. The
        `jok` flag indicates whether Jacobian data from previous calls can be
        reused (`jok = 1`) or if it must be recomputed (`jok = 0`). The
        `jnew` input is a single-item list that the user must control to tell
        the solve whether the Jacobian data has been updated (`jnew[0] = 1`)
        or not (`jnew[0] = 0`). An outlined example is given below.

        .. code-block:: python

            def psetupfn(t, y, yp, jok, jnew, gamma, userdata):
                if jok:
                    jnew[0] = 0
                else:
                    jnew[0] = 1
                    JJ = userdata['JJ']
                    JJ[:,:] = approx_jacobian(...)


            def psolvefn(t, y, yp, rvec, zvec, gamma, delta, lr, userdata):
                Pmat = np.eye(y.size) - gamma*userdata['JJ']
                zvec[:] = ...

        If you need additional information about CVODE preconditioners, please
        reference the original `SUNDIALS documentation`_.

        .. _SUNDIALS documentation: https://sundials.readthedocs.io/en/v6.1.1/ \
            cvode/Usage/index.html#preconditioner-solve-iterative-linear-solvers

        """

        if setupfn is None:
            pass
        elif not isinstance(setupfn, Callable):
            raise TypeError("'setupfn' must be type Callable.")

        if not isinstance(solvefn, Callable):
            raise TypeError("'solvefn' must be type Callable.")

        if side not in {'left', 'right', 'both'}:
            raise ValueError("'side' must be in {'left', 'right', 'both'}.")

        self.setupfn = setupfn
        self.solvefn = solvefn
        self.side = side

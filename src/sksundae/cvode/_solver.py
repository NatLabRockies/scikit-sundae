# cvode._solver.py

from __future__ import annotations

from typing import Callable, Literal, TYPE_CHECKING

from sksundae._cy_cvode import CVODE as _CVODE, CVODEResult as _CVODEResult

if TYPE_CHECKING:  # pragma: no cover
    from numpy import ndarray


class CVODE:
    """SUNDIALS CVODE solver."""

    def __init__(self, rhsfn: Callable, **options) -> None:
        """
        A class to wrap the C-based variable-coefficient ordinary differential
        equations (CVODE) solver from SUNDIALS [1]_ [2]_.

        Parameters
        ----------
        rhsfn : Callable
            Right-hand-side function with signature `f(t, y, yp[, userdata])`.
            See the notes for more information.
        **options : dict, optional
            Keyword arguments to describe the solver options. A full list of
            names, types, descriptions, and defaults is given below.
        userdata : object or None, optional
            Additional data object to supply to all user-defined callables.
            Cannot be None (default) if 'rhsfn' takes in 4 arguments.
        method : {'Adams', 'BDF'}, optional
            Specifies the linear multistep method. It is suggested to use 'BDF'
            (default) for stiff problems and 'Adams' for nonstiff problems.
        first_step : float, optional
            The initial step size. The default is 0, which uses an estimated
            value internally determined by SUNDIALS.
        min_step : float, optional
            Minimum allowable step size. The default is 0.
        max_step : float, optional
            Maximum allowable step size. Use 0 (default) for unbounded steps.
        rtol : float, optional
            Relative tolerance. It is recommended to not use values larger than
            1e-3 or smaller than 1e-15. The default is 1e-5.
        atol : float or array_like[float], optional
            Absolute tolerance. A scalar will apply to all variables equally,
            while an array (matching 'y' length) sets specific tolerances for
            eqch variable. The default is 1e-6.
        linsolver : {'dense', 'band', 'sparse', ...}, optional
            Choice of linear solver, defaults to 'dense'. 'band' requires both
            'lband' and 'uband'. 'sparse' uses SuperLU_MT [3]_ and requires
            'sparsity'. When using an iterative method ('gmres', 'bicgstab',
            'tfqmr') the number of Krylov dimensions is set using 'krylov_dim'.
            'lapackdense' and 'lapackband' can also be used as alternatives to
            'dense' and 'band'. They use OpenBLAS-linked LAPACK [4]_ routines,
            but can have noticeable overhead for small (<100) systems.
        lband : int or None, optional
            Lower Jacobian bandwidth. Given an ODE system `yp = f(t, y)`,
            the Jacobian is `J = df_i/dy_j`. Required when 'linsolver' is
            'band'. Use zero if no values are below the main diagonal. Defaults
            to None.
        uband : int or None, optional
            Upper Jacobian bandwidth. Required when 'linsolver' is 'band'. Use
            zero if no elements are above the main diagonal. Defaults to None.
        sparsity : 2D np.array or sparse matrix or None, optional
            Jacobian sparsity pattern. Required when 'linsolver' is 'sparse'.
            The shape must be (N, N) where N is the size of the system. Zero
            entries indicate fixed zeros in the Jacobian. If 'jacfn' is None,
            this argument activates a custom Jacobian routine (not part of the
            original SUNDIALS package). The routine works with all direct linear
            solvers but may increase step count. Reduce 'max_step' to help with
            this, if needed. Defaults to None.
        nthreads : int or None, optional
            Number of threads to use with the 'sparse' linear solver. If None
            (default), 1 is used. Use -1 to use all available threads.
        krylov_dim : int or None, optional
            Maximum number of Krylov basis vectors for iterative solvers. Will
            default to 5 if invalid/None when required. Larger values improve
            convergence but increase memory usage. Only applies to the 'gmres',
            'bicgstab', and 'tfqmr' linear solvers.
        max_order : int, optional
            Specifies the maximum order for the linear multistep method. BDF
            and Adams allow values in ranges [1, 5] and [1, 12], respectively.
            The default is the method's max, i.e., 5 for BDF and 12 for Adams.
        max_num_steps : int, optional
            The maximum number of steps taken by the solver in each attempt to
            reach the next output time. The default is 500.
        max_nonlin_iters : int, optional
            Specifies the maximum number of nonlinear solver iterations in one
            step. The default is 3.
        max_conv_fails : int, optional
            Specifies the max number of nonlinear solver convergence failures
            in one step. The default is 10.
        constraints_idx : array_like[int] or None, optional
            Specifies indices 'i' in the 'y' state variable array for which
            inequality constraints should be applied. Constraint types must be
            specified in 'constraints_type', see below. The default is None.
        constraints_type : array_like[int] or None, optional
            If 'constraints_idx' is not None, then this option must include an
            array of equal length specifying the types of constraints to apply.
            Values should be in `{-2, -1, 1, 2}` which apply `y[i] < 0`,
            `y[i] <= 0`, `y[i] >=0,` and `y[i] > 0`, respectively. The
            default is None.
        eventsfn : Callable or None, optional
            Events function with signature `g(t, y, events[, userdata])`.
            If None (default), no events are tracked. See the notes for more
            information. Requires 'num_events' be set when not None.

            The function may also have these optional attributes:

                terminal: list[bool, int], optional
                    Specifies solver behavior for each event. A boolean stops
                    the solver (True) or just records the event (False). An
                    integer stops the solver after than many occurrences. The
                    default is `[True]*num_events`.
                direction: list[int], optional
                    Determines which event slopes to track: `-1` (negative),
                    `1` (positive), or `0` (both). If not provided the
                    default `[0]*num_events` is used.

            You can assign attributes like `eventsfn.terminal = [True]` to
            any function in Python, after it has been defined.
        num_events : int, optional
            Number of events to track. The default is 0.
        jacfn : Callable or None, optional
            Jacobian function like `J(t, y, yp, JJ[, userdata])`. Fills the
            pre-allocated 2D matrix 'JJ' with values defined by the Jacobian
            `JJ[i,j] = dyp_i/dy_j`. An internal finite difference method is
            applied when None (default).
        precond : CVODEPrecond or None, optional
            Preconditioner functions. Only compatible with iterative linear
            solvers. Must be an instance of CVODEPrecond if not None (default).
        jactimes : CVODEJacTimes or None, optional
            Jacobian-vector product functions. Only compatible with iterative
            linear solvers. Must be an instance of CVODEJacTimes when provided.
            Difference quotient approximations are used with iterative solvers
            if None (default).

        Notes
        -----
        Return values from user-defined functions (e.g., 'rhsfn', 'eventsfn',
        and 'jacfn') are ignored by the solver. Instead the solver directly
        reads from pre-allocated memory. Output arrays (e.g., 'yp', 'events',
        and 'JJ') from each user-defined callable should be filled within each
        respective function. When setting values across the entire array/matrix
        at once, don't forget to use `[:]` to fill the existing array rather
        than overwriting it. For example, using `yp[:] = f(t, y)` is correct
        whereas `yp = f(t, y)` is not.

        When any user-defined function require data outside of their normal
        arguments, you can supply optional 'userdata'. When given, 'userdata'
        must appear in ALL function signatures ('rhsfn', 'eventsfn', 'jacfn',
        'precond', and 'jactimes') even if it is not used in all functions. Of
        course this does not apply to functions that are provided as `None`.
        Note that 'userdata' only takes up one argument position; however,
        'userdata' can be any Python object. Therefore, to pass more than one
        extra argument you should pack all data into a single tuple, dict,
        dataclass, etc. and pass them all together as 'userdata'. The data can
        be unpacked as needed within the functions.

        References
        ----------
        .. [1] A. C. Hindmarsh, P. N. Brown, K. E. Grant, S. L. Lee, R.
           Serban, D. E. Shumaker, and C. S. Woodward, "SUNDIALS: Suite of
           Nonlinear and Differential/Algebraic Equation Solvers," ACM TOMS,
           2005, DOI: 10.1145/1089014.1089020
        .. [2] D. J. Gardner, D. R. Reynolds, C. S. Woodward, C. J. Balos,
           "Enabling new flexibility in the SUNDIALS suite of nonlinear and
           differential/algebraic equation solvers," ACM TOMS, 2022,
           DOI: 10.1145/3539801
        .. [3] J. W. Demmel, J. R. Gilbert, and X. S. Li, "An Asynchronous
           Parallel Supernodal Algorithm for Sparse Gaussian Elimination,"
           SIMAX, 1999, DOI: 10.1137/S0895479897317685
        .. [4] E. Anderson, Z. Bai, C. Bischof, S. Blackford, J. Demmel, J.
           Dongarra, J. Du Croz, A. Greenbaum, S. Hammarling, A. McKenney, D.
           Sorensen, "LAPACK Users' Guide," Society for Industrial and Applied
           Mathematics, 1999, Philidelphia, PA.

        Examples
        --------
        The following example solves the stiff Van der Pol equation, a classic
        ODE test problem. The same example is provided by `MATLAB <VDP-Ex_>`_
        for comparison.

        .. _VDP-Ex:
            https://www.mathworks.com/help/matlab/math/solve-stiff-odes.html

        .. code-block:: python

            import numpy as np
            import sksundae as sun
            import matplotlib.pyplot as plt

            def rhsfn(t, y, yp):
                yp[0] =  y[1]
                yp[1] = 1000.*(1. - y[0]**2)*y[1] - y[0]

            solver = sun.cvode.CVODE(rhsfn)

            tspan = np.array([0, 3000])
            y0 = np.array([2, 0])

            soln = solver.solve(tspan, y0)

            plt.plot(soln.t, soln.y[:,0])
            plt.show()

        """
        self.__init_data = (rhsfn, options)
        self.__CVODE = _CVODE(rhsfn, **options)

    def __reduce__(self) -> tuple[type, tuple[Callable], dict]:
        """Custom pickling support due to C-extension."""
        return (_deserialize_cvode, self.__init_data)

    def init_step(self, t0: float, y0: ndarray) -> CVODEResult:
        """
        Initialize the solver.

        This method is called automatically when using 'solve'. However, it
        must be run manually, before the 'step' method, when solving with a
        step-by-step approach.

        Parameters
        ----------
        t0 : float
            Initial value of time.
        y0 : array_like[float], shape(m,)
            State variable values at 't0'. The length should match the number
            of equations in 'rhsfn'.

        Returns
        -------
        :class:`~sksundae.cvode.CVODEResult`
            Custom output class for CVODE solutions. Includes pretty-printing
            consistent with scipy outputs. See the class definition for more
            information.

        Raises
        ------
        MemoryError
            Failed to allocate memory for the CVODE solver.
        RuntimeError
            A SUNDIALS function returned NULL or was unsuccessful.

        """
        return self.__CVODE.init_step(t0, y0)

    def step(self, t: float, method: Literal['normal', 'onestep'] = 'normal',
             tstop: float | None = None) -> CVODEResult:
        """
        Return the solution at time 't'.

        Before calling the 'step' method, you must first initialize the solver
        by running 'init_step'.

        Parameters
        ----------
        t : float
            Value of time.
        method : {'normal', 'onestep'}, optional
            Solve method for the current step. When 'normal' (default), output
            is returned at time 't'. If 'onestep', output is returned after one
            internal step toward 't'. Both methods stop at events, if given,
            regardless of how 'eventsfn.terminal' was set.
        tstop : float or None, optional
            Specifies a hard time constraint for which the solver should not
            pass, regardless of the 'method'. The default is None.

        Returns
        -------
        :class:`~sksundae.cvode.CVODEResult`
            Custom output class for CVODE solutions. Includes pretty-printing
            consistent with scipy outputs. See the class definition for more
            information.

        Raises
        ------
        ValueError
            'method' value is invalid. Must be 'normal' or 'onestep'.
        ValueError
            'init_step' must be run prior to 'step'.

        Notes
        -----
        In general, when solving step by step, times should all be provided in
        either increasing or decreasing order. The solver can output results at
        times taken in the opposite direction of integration if the requested
        time is within the last internal step interval; however, values outside
        this interval will raise errors. Rather than trying to mix forward and
        reverse directions, choose each sequential time step carefully so you
        get all of the values you need.

        SUNDIALS provides a convenient graphic to help users understand how the
        step method and optional 'tstop' affect where the integrator stops. To
        read more, see their documentation `here`_.

        .. _here: https://computing.llnl.gov/projects/sundials/usage-notes

        """
        return self.__CVODE.step(t, method, tstop)

    def solve(self, tspan: ndarray, y0: ndarray) -> CVODEResult:
        """
        Return the solution across 'tspan'.

        Parameters
        ----------
        tspan : array_like[float], shape(n >= 2,)
            Solution time span. If `len(tspan) == 2`, the solution will be
            saved at internally chosen steps. When `len(tspan) > 2`, the
            solution saves the output at each specified time.
        y0 : array_like[float], shape(m,)
            State variable values at 'tspan[0]'. The length should match the
            number of equations in 'rhsfn'.

        Returns
        -------
        :class:`~sksundae.cvode.CVODEResult`
            Custom output class for CVODE solutions. Includes pretty-printing
            consistent with scipy outputs. See the class definition for more
            information.

        Raises
        ------
        ValueError
            'tspan' must be strictly increasing or decreasing.
        ValueError
            'tspan' length must be >= 2.

        """
        return self.__CVODE.solve(tspan, y0)


class CVODEResult(_CVODEResult):
    """Results container."""

    def __init__(self, **kwargs) -> None:
        """
        Inherits from :class:`~sksundae.common.RichResult`. The solution class
        groups output from :class:`CVODE` into an object. Descriptions of the
        fields are given below.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for the result fields. The full list of fields is
            given below.
        message : str
            Human-readable description of the status value.
        success : bool
            True if the solver was successful (status >= 0). False otherwise.
        status : int
            Reason for the algorithm termination. Negative values correspond
            to errors, and non-negative values to different successful criteria.
        t : ndarray, shape(n,)
            Solution time(s). The dimension depends on the method. Stepwise
            solutions will only have 1 value whereas solutions across a full
            'tspan' will have many.
        y : ndarray, shape(n, m)
            State variable values at each solution time. Rows correspond to
            indices in 't' and columns match indexing from 'y0'.
        i_events : ndarray, shape(k, num_events) or None
            Provides an array for each detected event 'k' specifying indices
            for which event(s) occurred. `i_events[k,i] != 0` if 'events[i]'
            occurred at 't_events[k]'. The sign of 'i_events' indicates the
            direction of zero-crossing:

                * -1 indicates 'events[i]' was decreasing
                * +1 indicates 'events[i]' was increasing

            Output for 'i_events' will be None when either 'eventsfn' was None
            or if no events occurred during the solve.
        t_events : ndarray, shape(k,) or None
            Times at which events occurred or None if 'eventsfn' was None or
            no events were triggered during the solve.
        y_events : ndarray, shape(k, m) or None
            State variable values at each 't_events' value or None. Rows and
            columns correspond to 't_events' and 'y0' indexing, respectively.
        nfev : int
            Number of times that 'rhsfn' was evaluated.
        njev : int
            Number of times the Jacobian was evaluated, 'jacfn' or internal
            finite difference method.

        Notes
        -----
        Terminal events are appended to the end of 't' and 'y'. However, if an
        event was not terminal then it will only appear in `*_events` outputs
        and not within the main output arrays.

        'nfev' and 'njev' are cumulative for stepwise solution approaches. The
        values are reset each time 'init_step' is called.

        """
        super().__init__(**kwargs)


def _deserialize_cvode(rhsfn: Callable, options: dict) -> CVODE:
    """Helper function for unpickling CVODE objects."""
    return CVODE(rhsfn, **options)

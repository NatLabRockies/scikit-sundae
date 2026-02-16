"""
Routines for analyzing Jacobian patterns and structures for problems involving
systems of differential-algebraic equations. For example, finding the sparsity
pattern and/or bandwidth.

"""

from __future__ import annotations
from typing import Callable, Any, TYPE_CHECKING

import inspect
from warnings import warn

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ndarray
    from scipy.sparse import spmatrix


def _cvode_pattern(rhsfn: Callable, t0: float, y0: ndarray,
                   userdata: Any = None) -> ndarray:
    """Jacobian pattern for CVODE functions. Access via j_pattern()."""
    # wrap rhsfn for cases w/ and w/o userdata
    signature = inspect.signature(rhsfn)

    if len(signature.parameters) == 3:
        def wrapper(t, y, yp): return rhsfn(t, y, yp)
    elif len(signature.parameters) == 4:
        if userdata is None:
            warn("'rhsfn' signature has 4 inputs so 'userdata' is expected,"
                 " but 'userdata=None'. Ensure this was intentional.")

        def wrapper(t, y, yp): return rhsfn(t, y, yp, userdata)
    else:
        raise ValueError("'rhsfn' signature must have either 3 or 4 inputs.")

    # recommended minimum perturbation
    dtype = y0.dtype if np.issubdtype(y0.dtype, np.floating) else np.float64
    uround = np.finfo(dtype).eps
    srur = np.sqrt(uround)

    # perturbed variables
    sign_y = (y0 >= 0).astype(float) * 2 - 1
    y = sign_y * np.maximum(uround, np.abs(y0))

    # initial derivatives
    yp_0 = np.zeros_like(y)
    wrapper(t0, y, yp_0)

    norm = max(srur, np.abs(yp_0).max())
    yp_0 = yp_0 / norm

    # Jacobian pattern
    def j_pattern(j):

        y_store = y[j]
        sign = (y[j] >= 0).astype(float) * 2 - 1

        y[j] += sign * srur * max(1.0, abs(y[j]))

        yp = np.zeros_like(y)
        wrapper(t0, y, yp)
        yp = yp / norm

        y[j] = y_store

        return (yp_0 != yp).astype(float)

    j_cols = [j_pattern(j) for j in range(y.size)]

    return np.column_stack(j_cols)


def _ida_pattern(resfn: Callable, t0: float, y0: ndarray, yp0: ndarray = None,
                 userdata: Any = None) -> ndarray:
    """Jacobian pattern for IDA functions. Access via j_pattern()."""
    # wrap resfn for cases w/ and w/o userdata
    signature = inspect.signature(resfn)

    if len(signature.parameters) == 4:
        def wrapper(t, y, yp, res): return resfn(t, y, yp, res)
    elif len(signature.parameters) == 5:
        if userdata is None:
            warn("'rhsfn' signature has 5 inputs so 'userdata' is expected,"
                 " but 'userdata=None'. Ensure this was intentional.")

        def wrapper(t, y, yp, res): return resfn(t, y, yp, res, userdata)
    else:
        raise ValueError("'rhsfn' signature must have either 4 or 5 inputs.")

    # recommended minimum perturbation
    dtype = y0.dtype if np.issubdtype(y0.dtype, np.floating) else np.float64
    uround = np.finfo(dtype).eps
    srur = np.sqrt(uround)

    # perturbed variables
    sign_y = (y0 >= 0).astype(float) * 2 - 1
    y = sign_y * np.maximum(uround, np.abs(y0))

    sign_yp = (yp0 >= 0).astype(float) * 2 - 1
    yp = sign_yp * np.maximum(uround, np.abs(yp0))

    # initial residuals
    res_0 = np.zeros_like(y)
    wrapper(t0, y, yp, res_0)

    norm = max(srur, np.abs(res_0).max())
    res_0 = res_0 / norm

    # Jacobian pattern
    rng = np.random.default_rng(42)
    rand = rng.random(2)

    def j_pattern(j):

        y_store, yp_store = y[j], yp[j]
        sign = (yp[j] >= 0).astype(float) * 2 - 1

        y[j] += sign * srur * rand[0] * max(1.0, abs(y[j]))
        yp[j] += sign * srur * rand[1] * max(1.0, abs(yp[j]))

        res = np.zeros_like(y)
        wrapper(t0, y, yp, res)
        res = res / norm

        y[j], yp[j] = y_store, yp_store

        return (res_0 != res).astype(float)

    j_cols = [j_pattern(j) for j in range(y.size)]

    return np.column_stack(j_cols)


def j_pattern(rhsfn: Callable, t0: float, y0: ndarray, yp0: ndarray = None,
              userdata: Any = None) -> ndarray:
    """
    Approximate the Jacobian pattern.

    This function uses a numerical Jacobian approximation for `rhsfn` about
    the given point to determine the Jacobian pattern. It requires evaluating
    the given function `N` times based on the size of `y0` so it can be
    slow for large systems.

    Be aware that this routine may return zeros in locations where ones should
    be depending on the evaluation point `y0` (and `yp0`). It is left to
    the user to determine a representative evaluation point that ensures all
    relevant relationships are correctly determined.

    Parameters
    ----------
    rhsfn : Callable
        Right-hand-side function for either an :class:`~sksundae.ida.IDA` or
        :class:`~sksundae.cvode.CVODE` problem. Signatures vary, see the class
        docstrings for more info. The correct routine is automatically selected
        depending on whether or not `yp0` is given. IDA requires that `yp0`
        be given and CVODE expects it to be None.
    t0 : float
        Input time to use in 'rhsfn'.
    y0 : ndarray
        State variables to use in 'rhsfn'.
    yp0 : ndarray or None, optional
        State variable time derivatives to use in 'rhsfn', by default None.
    userdata : Any, optional
        Additional data to pass to 'rhsfn', by default None.

    Returns
    -------
    pattern : 2D np.array
        Jacobian pattern represented by a 2D numpy array with shape (N, N).
        Ones or zeros in the position `A[i, j]` mean that function `F_i`
        either is or is not dependedent on variable `y_j`, respectively.

    """
    if yp0 is None:
        y0 = np.asarray(y0, dtype=float)
        return _cvode_pattern(rhsfn, t0, y0, userdata)
    else:
        y0 = np.asarray(y0, dtype=float)
        yp0 = np.asarray(yp0, dtype=float)
        return _ida_pattern(rhsfn, t0, y0, yp0, userdata)


def bandwidth(A: ndarray) -> tuple[int]:
    """
    Return half bandwidths of a 2D array.

    Uses the `scipy.linalg.bandwidth` function to determine the lower and
    upper bandwidths of a given array. Use in conjunction with `j_pattern`
    to find the bandwidths of a Jacobian pattern.

    Parameters
    ----------
    A : 2D np.array
        Input array of size (N, M).

    Returns
    -------
    bands : tuple[int]
        A 2-tuple of integers indicating the lower and upper half bandwidths
        of the given matrix. A zero denotes that there are no non-zero elements
        on that side. The full bandwidth is `lband + uband + 1`.

    """
    from scipy.linalg import bandwidth
    return bandwidth(A)


def reduce_bandwidth(A: ndarray | spmatrix,
                     symmetric: bool = False) -> tuple[ndarray]:
    """
    Find a row/col reordering to reduce bandwidth.

    Uses the Reverse Cuthill-McKee algorithm from `scipy.sparse.csgraph` to
    determine an index rearragement for rows and columns of `A` that reduce
    the bandwidth.

    Parameters
    ----------
    A : ndarray | spmatrix
        A 2D (n, n) input matrix whose sparsity pattern will be reduced.
    symmetric : bool, optional
        True if input matrix is guaranteed symmetric, otherwise False (default).

    Returns
    -------
    perm : ndarray
        Array of permuted row and column indices such that B = A[perm][:, perm]
        is a maxtrix with reduced bandwidth compared to A.
    inv_perm : ndarray
        The inverse of 'perm' such that B[inv_perm][:, inv_perm] returns the
        original matrix A.

    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    if not sp.issparse(A):
        A = sp.csc_matrix(A)
    else:
        A = A.tocsc()

    perm = reverse_cuthill_mckee(A, symmetric)
    inv_perm = np.argsort(perm)

    return perm, inv_perm

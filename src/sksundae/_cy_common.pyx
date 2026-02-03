# _cy_common.pyx

# Dependencies
cimport numpy as np

# Extern cdef headers
from .c_sundials cimport *  # Access to C types
from .c_nvector cimport *  # Access to N_Vector functions
from .c_sunmatrix cimport *  # Access to SUNMatrix functions

# Define float and int types:
# py_config.pxi is created in setup.py. While building the python package, the 
# sundials_config.h header is parsed to determine what precision was used to
# compile the SUNDIALS that is being built against. The settings are saved in
# the pxi file and used here.
include "py_config.pxi"

config = {
    "SUNDIALS_VERSION": SUNDIALS_VERSION,
    "SUNDIALS_FLOAT_TYPE": SUNDIALS_FLOAT_TYPE,
    "SUNDIALS_INT_TYPE": SUNDIALS_INT_TYPE,
    "SUNDIALS_SUPERLUMT_ENABLED": SUNDIALS_SUPERLUMT_ENABLED,
    "SUNDIALS_SUPERLUMT_THREAD_TYPE": SUNDIALS_SUPERLUMT_THREAD_TYPE,
    "SUNDIALS_BLAS_LAPACK_ENABLED": SUNDIALS_BLAS_LAPACK_ENABLED,
}

if SUNDIALS_FLOAT_TYPE == "float":
    from numpy import float32 as DTYPE
elif SUNDIALS_FLOAT_TYPE == "double":
    from numpy import float64 as DTYPE
elif SUNDIALS_FLOAT_TYPE == "long double":
    from numpy import longdouble as DTYPE

if SUNDIALS_INT_TYPE == "int":
    from numpy import int32 as INT_TYPE
elif SUNDIALS_INT_TYPE == "long int":
    from numpy import int64 as INT_TYPE


cdef svec2np(N_Vector nvec, np.ndarray[DTYPE_t, ndim=1] np_array):
    """Fill a numpy array with values from an N_Vector."""
    cdef sunrealtype* nv_ptr

    nv_ptr = N_VGetArrayPointer(nvec)
    ptr2np(nv_ptr, np_array)


cdef np2svec(np.ndarray[DTYPE_t, ndim=1] np_array, N_Vector nvec):
    """Fill an N_Vector with values from a numpy array."""
    cdef sunrealtype* nv_ptr

    nv_ptr = N_VGetArrayPointer(nvec)
    np2ptr(np_array, nv_ptr)


cdef ptr2np(sunrealtype* nv_ptr, np.ndarray[DTYPE_t, ndim=1] np_array):
    """Fill a numpy array with values from an N_Vector pointer."""
    cdef sunindextype size = <sunindextype> np_array.size

    np_array[:] = <sunrealtype[:size]> nv_ptr


cdef np2ptr(np.ndarray[DTYPE_t, ndim=1] np_array, sunrealtype* nv_ptr):
    """Fill an N_Vector pointer with values from a numpy array."""
    cdef sunindextype size = <sunindextype> np_array.size

    nv_ptr[0:size] = &np_array[0]


cdef np2smat_dense(np.ndarray[DTYPE_t, ndim=2] np_A, SUNMatrix smat):
    """Fill a SUNDenseMatrix with values from a 2D numpy array."""
    cdef sunindextype i, j
    cdef sunindextype M = <sunindextype> np_A.shape[0]
    cdef sunindextype N = <sunindextype> np_A.shape[1]
    cdef sunrealtype** sm_cols = SUNDenseMatrix_Cols(smat)

    for j in range(N):
        for i in range(M):
            sm_cols[j][i] = np_A[i,j]


cdef np2smat_band(np.ndarray[DTYPE_t, ndim=2] np_A, SUNMatrix smat):
    """Fill a SUNBandMatrix with values from a 2D numpy array."""
    cdef sunindextype i, j
    cdef sunindextype N = <sunindextype> np_A.shape[1]
    cdef sunrealtype** sm_cols = SUNBandMatrix_Cols(smat)
    cdef sunindextype lband = SUNBandMatrix_LowerBandwidth(smat)
    cdef sunindextype uband = SUNBandMatrix_UpperBandwidth(smat)
    cdef sunindextype smu = SUNBandMatrix_StoredUpperBandwidth(smat)

    # Indexing is more complex in a SUNBandMatrix, see documentation:
    # https://sundials.readthedocs.io/en/latest/sunmatrix/SUNMatrix_links.html
    for j in range(N):
        i_min = max(0, j - uband)
        i_max = min(N, j + lband + 1)
        for i in range(i_min, i_max):
            sm_cols[j][i-j+smu] = np_A[i,j]


cdef np2smat_sparse1D(np.ndarray[DTYPE_t, ndim=1] np_A, SUNMatrix smat,
                      object sparsity):
    """Fill a SUNSparseMatrix with values from a 1D numpy array."""
    cdef sunindextype nnz, nidx, nptr
    cdef sunrealtype* data = SUNSparseMatrix_Data(smat)
    cdef sunindextype* indices = SUNSparseMatrix_IndexValues(smat)
    cdef sunindextype* indptrs = SUNSparseMatrix_IndexPointers(smat)
    cdef np.ndarray[INT_TYPE_t, ndim=1] np_indices = sparsity.indices
    cdef np.ndarray[INT_TYPE_t, ndim=1] np_indptr = sparsity.indptr

    nnz = <sunindextype> sparsity.nnz
    nidx = <sunindextype> sparsity.indices.size
    nptr = <sunindextype> sparsity.indptr.size

    data[0:nnz] = &np_A[0]
    indices[0:nidx] = <sunindextype*> &np_indices[0]
    indptrs[0:nptr] = <sunindextype*> &np_indptr[0]


cdef np2smat_sparse2D(np.ndarray[DTYPE_t, ndim=2] np_A, SUNMatrix smat,
                      object sparsity):
    """Fill a SUNSparseMatrix with values from a 2D numpy array."""
    cdef sunindextype i, j, start, end, idx
    cdef sunindextype nnz, nidx, nptr, ncols
    cdef sunrealtype* data = SUNSparseMatrix_Data(smat)
    cdef sunindextype* indices = SUNSparseMatrix_IndexValues(smat)
    cdef sunindextype* indptrs = SUNSparseMatrix_IndexPointers(smat)
    cdef np.ndarray[INT_TYPE_t, ndim=1] np_indices = sparsity.indices
    cdef np.ndarray[INT_TYPE_t, ndim=1] np_indptr = sparsity.indptr

    nidx = <sunindextype> sparsity.indices.size
    nptr = <sunindextype> sparsity.indptr.size

    indices[0:nidx] = <sunindextype*> &np_indices[0]
    indptrs[0:nptr] = <sunindextype*> &np_indptr[0]

    ncols = <sunindextype> np_A.shape[1]
    for j in range(ncols):
        start = sparsity.indptr[j]
        end = sparsity.indptr[j+1]

        for i, idx in enumerate(sparsity.indices[start:end]):
            data[start + i] = np_A[idx,j]


cdef np2smat(np.ndarray np_A, SUNMatrix smat, object sparsity):
    """Fill a SUNMatrix with values from np_A using the correct cdef."""
    cdef SUNMatrix_ID matrix_id = SUNMatGetID(smat)

    if matrix_id == SUNMATRIX_DENSE:
        np2smat_dense(np_A, smat)
    elif matrix_id == SUNMATRIX_BAND:
        np2smat_band(np_A, smat)
    elif matrix_id == SUNMATRIX_SPARSE and np_A.ndim == 1:
        np2smat_sparse1D(np_A, smat, sparsity)
    elif matrix_id == SUNMATRIX_SPARSE and np_A.ndim == 2:
        np2smat_sparse2D(np_A, smat, sparsity)
    else:
        raise TypeError("Only 'dense', 'band', or 'sparse' SUNMatrix are"
                        " supported for 'smat'.")

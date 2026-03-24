# _cy_common.pxd

# Dependencies
cimport numpy as np

# Extern cdef headers
from .c_sundials cimport *  # Access to C types

# Propagate python exceptions or print SUNDIALS error messages
cdef _pyerr_handler()
cdef void _sunerr_handler(
    int line, const char* func, const char* file, const char* msg, int err_code,
    void* err_user_data, SUNContext ctx) except *

# Convert between N_Vector and numpy array
cdef svec2np(N_Vector nvec, np.ndarray[DTYPE_t, ndim=1] np_array)
cdef np2svec(np.ndarray[DTYPE_t, ndim=1] np_array, N_Vector nvec)

# Convert between sunrealtype* and numpy array
cdef ptr2np(sunrealtype* nv_ptr, np.ndarray[DTYPE_t, ndim=1] np_array)
cdef np2ptr(np.ndarray[DTYPE_t, ndim=1] np_array, sunrealtype* nv_ptr)

# Fill SUNMatrrix with values from 2D numpy array
cdef np2smat(np.ndarray np_A, SUNMatrix smat, object sparsity)

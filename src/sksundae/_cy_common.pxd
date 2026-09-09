# Dependencies
cimport numpy as np

# Extern cdef headers
from .c_sundials cimport *  # Access to C types

# Propagate python exceptions or print SUNDIALS error messages
cdef _pyerr_handler()
cdef void _sunerr_handler(
    int line, const char* func, const char* file, const char* msg, int err_code,
    void* err_user_data, SUNContext ctx) except *

# Convert between N_Vector and numpy array. Arrays are read-only unless
# 'writable' is True, since most are direct views into SUNDIALS' own memory.
cdef np.ndarray[DTYPE_t, ndim=1] svec2np(N_Vector nvec, bint writable=*)
cdef np.ndarray[DTYPE_t, ndim=1] sptr2np(
    sunrealtype* nv_ptr, Py_ssize_t length, bint writable=*)

# Fill SUNMatrrix with values from 2D numpy array
cdef np2smat(np.ndarray np_A, SUNMatrix smat, object sparsity)

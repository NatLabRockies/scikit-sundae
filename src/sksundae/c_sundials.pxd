# Dependencies
cimport numpy as np

# Define float and int types:
# c_config.pxi is created in setup.py. While building the python package, the 
# sundials_config.h header is parsed to determine what precision was used to
# compile the SUNDIALS that is being built against. The settings are saved in
# the pxi file and used here.
include "c_config.pxi"

# sundials_types.h
cdef extern from "sundials/sundials_types.h":
    ctypedef struct _SUNContext:
        pass

    ctypedef int SUNComm
    ctypedef int sunbooleantype
    ctypedef _SUNContext* SUNContext

    ctypedef void (*SUNErrHandlerFn)(int line, const char* func, const char* file,
                                     const char* msg, int err_code, void* err_user_data,
                                     SUNContext ctx) except *

    int SUN_COMM_NULL

# sundials_context.h
cdef extern from "sundials/sundials_context.h":
    int SUNContext_Create(int comm, SUNContext* ctx)
    void SUNContext_Free(SUNContext* ctx)

    int SUNContext_ClearErrHandlers(SUNContext ctx)
    int SUNContext_PushErrHandler(SUNContext ctx, SUNErrHandlerFn err_fn,
                                  void* err_user_data)

# sundials_nvector.h
cdef extern from "sundials/sundials_nvector.h":
    ctypedef struct _N_Vector:
        pass
    ctypedef _N_Vector* N_Vector

    void N_VDestroy(N_Vector v)
    sunrealtype* N_VGetArrayPointer(N_Vector v)
    sunindextype N_VGetLength(N_Vector v)

# sundials_matrix.h
cdef extern from "sundials/sundials_matrix.h":
    ctypedef struct _SUNMatrix:
        pass 
    ctypedef _SUNMatrix* SUNMatrix

    ctypedef enum SUNMatrix_ID:
        SUNMATRIX_DENSE
        SUNMATRIX_BAND
        SUNMATRIX_SPARSE

    SUNMatrix_ID SUNMatGetID(SUNMatrix A)

    void SUNMatDestroy(SUNMatrix A)

# sundials_linearsolver.h
cdef extern from "sundials/sundials_linearsolver.h":
    ctypedef struct _SUNLinearSolver:
        pass 
    ctypedef _SUNLinearSolver* SUNLinearSolver

    int SUNLinSolFree(SUNLinearSolver LS)

# sundials_iterative.h
cdef extern from "sundials/sundials_iterative.h":
    cdef enum:
        SUN_PREC_NONE
        SUN_PREC_LEFT
        SUN_PREC_RIGHT
        SUN_PREC_BOTH

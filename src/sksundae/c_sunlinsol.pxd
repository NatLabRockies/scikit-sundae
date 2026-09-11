from .c_sundials cimport *  # Access to types

# sunlinsol_dense.h
cdef extern from "sunlinsol/sunlinsol_dense.h":
    SUNLinearSolver SUNLinSol_Dense(N_Vector y, SUNMatrix A, SUNContext ctx)

# sunlinsol_band.h
cdef extern from "sunlinsol/sunlinsol_band.h":
    SUNLinearSolver SUNLinSol_Band(N_Vector y, SUNMatrix A, SUNContext ctx)

# sunlinsol_spgmr.h
cdef extern from "sunlinsol/sunlinsol_spgmr.h":
    SUNLinearSolver SUNLinSol_SPGMR(N_Vector y, int pretype, int maxl,
                                    SUNContext ctx)

# sunlinsol_spbcgs.h
cdef extern from "sunlinsol/sunlinsol_spbcgs.h":
    SUNLinearSolver SUNLinSol_SPBCGS(N_Vector y, int pretype, int maxl,
                                     SUNContext ctx)

# sunlinsol_sptfqmr.h
cdef extern from "sunlinsol/sunlinsol_sptfqmr.h":
    SUNLinearSolver SUNLinSol_SPTFQMR(N_Vector y, int pretype, int maxl,
                                      SUNContext ctx)

# sunlinsol_superlumt.h - real or dummy, depending on availability
cdef extern from "./include/superlumt_wrapper.h":
    SUNLinearSolver SUNLinSol_SuperLUMT(N_Vector y, SUNMatrix A, int nthreads,
                                        SUNContext ctx)

# sunlinsol_lapackdense.h - real or dummy, depending on availability
cdef extern from "./include/lapackdense_wrapper.h":
    SUNLinearSolver SUNLinSol_LapackDense(N_Vector y, SUNMatrix A,
                                          SUNContext ctx)

# sunlinsol_lapackband.h - real or dummy, depending on availability
cdef extern from "./include/lapackband_wrapper.h":
    SUNLinearSolver SUNLinSol_LapackBand(N_Vector y, SUNMatrix A,
                                         SUNContext ctx)

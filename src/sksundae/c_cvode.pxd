# c_cvode.pxd

from .c_sundials cimport *  # Access to types

# cvode.h
cdef extern from "cvode/cvode.h":

    # user-supplied functions
    ctypedef int (*CVRhsFn)(sunrealtype t, N_Vector yy, N_Vector yp, void* data) except -1
    ctypedef int (*CVRootFn)(sunrealtype t, N_Vector yy, sunrealtype* ee, void* data) except -1

    # imethod
    int CV_ADAMS
    int CV_BDF
    
    # itask
    int CV_NORMAL
    int CV_ONE_STEP

    # return values
    int CV_SUCCESS
    int CV_TSTOP_RETURN
    int CV_ROOT_RETURN
    
    # initialization functions
    void* CVodeCreate(int imethod, SUNContext ctx)
    int CVodeInit(void* mem, CVRhsFn rhsfn, sunrealtype t0, N_Vector y0)
    int CVodeReInit(void* mem, sunrealtype t0, N_Vector y0)

    # tolerance input functions
    int CVodeSStolerances(void* mem, sunrealtype rtol, sunrealtype atol)
    int CVodeSVtolerances(void* mem, sunrealtype rtol, N_Vector atol)

    # optional input functions
    int CVodeSetUserData(void* mem, void* data)
    int CVodeSetMaxOrd(void* mem, int max_order)
    int CVodeSetMaxNumSteps(void* mem, long int max_num_steps)
    int CVodeSetInitStep(void* mem, sunrealtype first_step)
    int CVodeSetMaxStep(void* mem, sunrealtype max_step)
    int CVodeSetMinStep(void* mem, sunrealtype min_step)
    int CVodeSetStopTime(void* mem, sunrealtype tstop)
    int CVodeClearStopTime(void* mem)
    int CVodeSetConstraints(void* mem, N_Vector constraints)

    # nonlinear solver input functions
    int CVodeSetMaxConvFails(void* mem, int max_conv_fails)
    int CVodeSetMaxNonlinIters(void* mem, int max_nonlin_iters)

    # rootfinding initialization function
    int CVodeRootInit(void* mem, int nrtfn, CVRootFn eventsfn)

    # rootfinding optional input functions
    int CVodeSetRootDirection(void* mem, int* direction)

    # solver function
    int CVode(void* mem, sunrealtype tend, N_Vector yret, sunrealtype* tret, 
              int itask)
    
    # optional output functions
    int CVodeGetRootInfo(void* mem, int* rootsfound)
    int CVodeGetNumRhsEvals(void* mem, long int* nrevals)
    
    # free functions
    void CVodeFree(void** mem)

# cvode_ls.h
cdef extern from "cvode/cvode_ls.h":

    # user-supplied functions
    ctypedef int (*CVLsJacFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, SUNMatrix JJ, void* data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) except -1

    ctypedef int (*CVLsPrecSetupFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, sunbooleantype jok, sunbooleantype* jcurPtr,
        sunrealtype gamma, void* data) except -1

    ctypedef int (*CVLsPrecSolveFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, N_Vector rv, N_Vector zv, sunrealtype gamma,
        sunrealtype delta, int lr, void* data) except -1

    ctypedef int (*CVLsJacTimesSetupFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, void* data) except -1

    ctypedef int (*CVLsJacTimesVecFn)(
        N_Vector vv, N_Vector Jv, sunrealtype tt, N_Vector yy, N_Vector yp, void* data,
        N_Vector tmp) except -1

    # exported functions
    int CVodeSetLinearSolver(void* mem, SUNLinearSolver LS, SUNMatrix A)

    # optional inputs to LS interface
    int CVodeSetJacFn(void* mem, CVLsJacFn jacfn)
    int CVodeSetPreconditioner(void* mem, CVLsPrecSetupFn psetup, CVLsPrecSolveFn psolve)
    int CVodeSetJacTimes(void* mem, CVLsJacTimesSetupFn jvsetup, CVLsJacTimesVecFn jvsolve)

    # optional outputs from LS interface
    int CVodeGetNumJacEvals(void* mem, long int* njevals)

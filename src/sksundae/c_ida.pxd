# c_ida.pxd

from .c_sundials cimport *

# ida.h
cdef extern from "ida/ida.h":

    # user-supplied functions
    ctypedef int (*IDAResFn)(sunrealtype t, N_Vector yy, N_Vector yp, N_Vector rr, void* data) except -1
    ctypedef int (*IDARootFn)(sunrealtype t, N_Vector yy, N_Vector yp, sunrealtype* ee, void* data) except -1

    # itask
    int IDA_NORMAL
    int IDA_ONE_STEP

    # icopt
    int IDA_YA_YDP_INIT
    int IDA_Y_INIT

    # return values
    int IDA_SUCCESS
    int IDA_TSTOP_RETURN
    int IDA_ROOT_RETURN
    
    # initialization functions
    void* IDACreate(SUNContext ctx)
    int IDAInit(void* mem, IDAResFn resfn, sunrealtype t0, N_Vector y0, N_Vector yp0)
    int IDAReInit(void* mem, sunrealtype t0, N_Vector y0, N_Vector yp0)

    # tolerance input functions
    int IDASStolerances(void* mem, sunrealtype rtol, sunrealtype atol)
    int IDASVtolerances(void* mem, sunrealtype rtol, N_Vector atol)

    # initial condition calculation function
    int IDACalcIC(void* mem, int ic_opt, sunrealtype ic_t0)

    # optional input functions
    int IDASetUserData(void* mem, void* data)
    int IDASetMaxOrd(void* mem, int max_order)
    int IDASetMaxNumSteps(void* mem, long int max_num_steps)
    int IDASetInitStep(void* mem, sunrealtype first_step)
    int IDASetMaxStep(void* mem, sunrealtype max_step)
    int IDASetMinStep(void* mem, sunrealtype min_step)
    int IDASetStopTime(void* mem, sunrealtype tstop)
    int IDAClearStopTime(void* mem)
    int IDASetId(void* mem, N_Vector algidx)
    int IDASetConstraints(void* mem, N_Vector constraints)

    # nonlinear solver input functions
    int IDASetMaxConvFails(void* mem, int max_conv_fails)
    int IDASetMaxNonlinIters(void* mem, int max_nonlin_iters)

    # rootfinding initialization function
    int IDARootInit(void* mem, int nrtfn, IDARootFn eventsfn)

    # rootfinding optional input functions
    int IDASetRootDirection(void* mem, int* direction)

    # solver function
    int IDASolve(void* mem, sunrealtype tend, sunrealtype* tret, N_Vector yret,
                 N_Vector ypret, int itask)
    
    # optional output functions
    int IDAGetConsistentIC(void* mem, N_Vector yy0_mod, N_Vector yp0_mod)
    int IDAGetRootInfo(void* mem, int* rootsfound)
    int IDAGetNumResEvals(void* mem, long int* nrevals)
    int IDAGetCurrentStep(void* mem, sunrealtype* hcur)
    
    # free functions
    void IDAFree(void** mem)

# ida_ls.h
cdef extern from "ida/ida_ls.h":

    # user-supplied functions
    ctypedef int (*IDALsJacFn)(
        sunrealtype t, sunrealtype cj, N_Vector yy, N_Vector yp, N_Vector rr,
        SUNMatrix JJ, void* data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) except -1

    ctypedef int (*IDALsPrecSetupFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, N_Vector rr, sunrealtype cj,
        void* data) except -1

    ctypedef int (*IDALsPrecSolveFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, N_Vector rr, N_Vector rv,
        N_Vector zv, sunrealtype cj, sunrealtype delta, void* data) except -1

    ctypedef int (*IDALsJacTimesSetupFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, N_Vector rr, sunrealtype cj,
        void* data) except -1

    ctypedef int (*IDALsJacTimesVecFn)(
        sunrealtype tt, N_Vector yy, N_Vector yp, N_Vector rr, N_Vector vv, N_Vector Jv,
        sunrealtype cj, void* data, N_Vector tmp1, N_Vector tmp2) except -1

    # exported functions
    int IDASetLinearSolver(void* mem, SUNLinearSolver LS, SUNMatrix A)

    # optional inputs to LS interface
    int IDASetJacFn(void* mem, IDALsJacFn jacfn)
    int IDASetPreconditioner(void* mem, IDALsPrecSetupFn psetup, IDALsPrecSolveFn psolve)
    int IDASetJacTimes(void* mem, IDALsJacTimesSetupFn jvsetup, IDALsJacTimesVecFn jvsolve)

    # optional outputs from LS interface
    int IDAGetNumJacEvals(void* mem, long int* njevals)

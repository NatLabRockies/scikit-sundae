import pytest
import numpy as np
import numpy.testing as npt

from sksundae.cvode import CVODE, CVODEPrecond, CVODEJacTimes


def ode(t, y, yp):
    yp[0] = 0.1
    yp[1] = y[1]


def ode_soln(t, y0):
    if hasattr(t, 'size'):
        y = np.zeros([t.size, 2])
        y[:, 0] = 0.1*t + y0[0]
        y[:, 1] = y0[1]*np.exp(t)
    else:
        y = np.zeros([2])
        y[0] = 0.1*t + y0[0]
        y[1] = y0[1]*np.exp(t)
    return y


def rhsfn(t, y, yp, userdata):
    yp[0] = y[1]
    yp[1] = 1000*(1 - y[0]**2)*y[1] - y[0]


def jacfn(t, y, yp, JJ, userdata):
    JJ[0, 0] = 0
    JJ[0, 1] = 1
    JJ[1, 0] = -2000*y[0]*y[1] - 1
    JJ[1, 1] = 1000*(1 - y[0]**2)


def psetupfn(t, y, yp, jok, jnew, gamma, userdata):
    if jok:
        jnew[0] = 0
    else:
        jnew[0] = 1
        JJ = userdata['JJ']
        jacfn(t, y, yp, JJ, userdata)


def psolvefn(t, y, yp, rvec, zvec, gamma, delta, lr, userdata):
    Pmat = np.eye(y.size) - gamma*userdata['JJ']
    zvec[:] = np.linalg.solve(Pmat, rvec)


def jvode(t, y, yp, userdata):
    yp[0] = 0.1
    yp[1] = y[1]


def jvsetupfn(t, y, yp, userdata):
    JJ = userdata['JJ']
    JJ[:, :] = np.array([[0, 0], [0, 1]])


def jvsolvefn(t, y, yp, v, Jv, userdata):
    JJ = userdata['JJ']
    Jv[:] = JJ.dot(v)


@pytest.mark.parametrize('linsolver', ('gmres', 'bicgstab', 'tfqmr'))
def test_iterative_no_precond(linsolver):
    y0 = np.array([1, 2])

    solver = CVODE(ode, linsolver=linsolver, rtol=1e-9, atol=1e-12)

    tspan = np.linspace(0, 10, 11)  # normal solve - user picks times
    soln = solver.solve(tspan, y0)
    assert len(tspan) > 2 and len(tspan) == len(soln.t)
    npt.assert_allclose(soln.y, ode_soln(soln.t, y0), rtol=1e-5)

    tspan = np.array([0, 10])  # onestep solve - integrator picks times
    soln = solver.solve(tspan, y0)
    assert len(tspan) == 2 and len(soln.t) > 2
    npt.assert_allclose(soln.y, ode_soln(soln.t, y0), rtol=1e-5)


@pytest.mark.parametrize('linsolver', ('gmres', 'bicgstab', 'tfqmr'))
def test_incompatible_options(linsolver):

    def jacfn(t, y, yp, JJ):
        pass

    # iterative solvers don't support jacfn from sparsity or explicit
    with pytest.raises(ValueError):
        _ = CVODE(ode, linsolver=linsolver, sparisty=np.eye(2))

    with pytest.raises(ValueError):
        _ = CVODE(ode, linsolver=linsolver, jacfn=jacfn)


def test_preconditioner():

    # with psetupfn = None
    for side in ['left', 'right', 'both']:
        precond = CVODEPrecond(None, psolvefn, side=side)
        assert precond.setupfn is None
        assert callable(precond.solvefn)
        assert precond.side == side

    with pytest.raises(TypeError):
        _ = CVODEPrecond('psetupfn', psolvefn)

    with pytest.raises(ValueError):  # bad side
        _ = CVODEPrecond(None, psolvefn, side='fake')

    # with psetupfn defined
    for side in ['left', 'right', 'both']:
        precond = CVODEPrecond(psetupfn, psolvefn, side)
        assert callable(precond.setupfn)
        assert callable(precond.solvefn)
        assert precond.side == side

    with pytest.raises(TypeError):
        _ = CVODEPrecond(psetupfn, 'psolvefn')

    with pytest.raises(ValueError):  # bad side
        _ = CVODEPrecond(psetupfn, psolvefn, 'fake')

    # accidentally switching psolvefn and psetupfn
    precond = CVODEPrecond(psolvefn, psetupfn)
    with pytest.raises(ValueError):
        _ = CVODE(rhsfn, linsolver='gmres', precond=precond,
                  userdata={})


@pytest.mark.parametrize(('linsolver', 'side'), (('gmres', 'left'),
                         ('bicgstab', 'right'), ('tfqmr', 'both')))
def test_w_precond_solve(linsolver, side):
    tspan = np.array([0, 3000])
    y0 = np.array([2, 0])

    precond = CVODEPrecond(psetupfn, psolvefn, side)
    userdata = {'JJ': np.zeros((y0.size, y0.size))}

    solver = CVODE(rhsfn, linsolver=linsolver, precond=precond,
                   userdata=userdata)

    soln = solver.solve(tspan, y0)
    assert soln.success


def test_jactimes():

    # with jvsetupfn = None
    jactimes = CVODEJacTimes(None, jvsolvefn)
    assert jactimes.setupfn is None
    assert callable(jactimes.solvefn)

    with pytest.raises(TypeError):
        _ = CVODEJacTimes('jvsetupfn', jvsolvefn)

    # with jvsetupfn defined
    jactimes = CVODEJacTimes(jvsetupfn, jvsolvefn)
    assert callable(jactimes.setupfn)
    assert callable(jactimes.solvefn)

    with pytest.raises(TypeError):
        _ = CVODEJacTimes(jvsetupfn, 'jvsolvefn')

    # accidentally switching jvsolvefn and jvsetupfn
    jactimes = CVODEJacTimes(jvsolvefn, jvsetupfn)
    with pytest.raises(ValueError):
        _ = CVODE(rhsfn, linsolver='gmres', jactimes=jactimes,
                  userdata={})


@pytest.mark.parametrize('linsolver', ('gmres', 'bicgstab', 'tfqmr'))
def test_w_jactimes_solve(linsolver):
    tspan = np.array([0, 10])
    y0 = np.array([1, 2])

    jactimes = CVODEJacTimes(jvsetupfn, jvsolvefn)
    userdata = {'JJ': np.zeros((y0.size, y0.size))}

    solver = CVODE(jvode, linsolver=linsolver, jactimes=jactimes,
                   userdata=userdata)

    soln = solver.solve(tspan, y0)
    assert soln.success


def test_readonly_arrays_precond():
    tspan = np.array([0, 3000])
    y0 = np.array([2, 0])
    userdata = {'JJ': np.zeros((y0.size, y0.size))}

    # y is read-only in psetupfn
    def bad_psetupfn_y(t, y, yp, jok, jnew, gamma, userdata):
        y[0] = 0.0

    precond = CVODEPrecond(bad_psetupfn_y, psolvefn)
    solver = CVODE(rhsfn, linsolver='gmres', precond=precond,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    # yp is read-only in psetupfn
    def bad_psetupfn_yp(t, y, yp, jok, jnew, gamma, userdata):
        yp[0] = 0.0

    precond = CVODEPrecond(bad_psetupfn_yp, psolvefn)
    solver = CVODE(rhsfn, linsolver='gmres', precond=precond,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    # y is read-only in psolvefn
    def bad_psolvefn_y(t, y, yp, rvec, zvec, gamma, delta, lr, userdata):
        y[0] = 0.0
        zvec[:] = rvec

    precond = CVODEPrecond(psetupfn, bad_psolvefn_y)
    solver = CVODE(rhsfn, linsolver='gmres', precond=precond,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    # yp is read-only in psolvefn
    def bad_psolvefn_yp(t, y, yp, rvec, zvec, gamma, delta, lr, userdata):
        yp[0] = 0.0
        zvec[:] = rvec

    precond = CVODEPrecond(psetupfn, bad_psolvefn_yp)
    solver = CVODE(rhsfn, linsolver='gmres', precond=precond,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    # rvec is read-only in psolvefn
    def bad_psolvefn_rvec(t, y, yp, rvec, zvec, gamma, delta, lr, userdata):
        rvec[0] = 0.0
        zvec[:] = rvec

    precond = CVODEPrecond(None, bad_psolvefn_rvec)
    solver = CVODE(rhsfn, linsolver='gmres', precond=precond,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)


def test_readonly_arrays_jactimes():
    tspan = np.array([0, 3000])
    y0 = np.array([2, 0])
    userdata = {'JJ': np.zeros((y0.size, y0.size))}

    # y is read-only in jvsetupfn
    def bad_jvsetupfn_y(t, y, yp, userdata):
        y[0] = 0.0

    jactimes = CVODEJacTimes(bad_jvsetupfn_y, jvsolvefn)
    solver = CVODE(rhsfn, linsolver='gmres', jactimes=jactimes,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    # yp is read-only in jvsetupfn
    def bad_jvsetupfn_yp(t, y, yp, userdata):
        yp[0] = 0.0

    jactimes = CVODEJacTimes(bad_jvsetupfn_yp, jvsolvefn)
    solver = CVODE(rhsfn, linsolver='gmres', jactimes=jactimes,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    # y, yp, and v are read-only in jvsolvefn - use the stiffer 'rhsfn'
    # problem so GMRES actually needs matrix-vector products (jvode is too
    # easy to trigger it)
    def jvsetupfn_hard(t, y, yp, userdata):
        jacfn(t, y, yp, userdata['JJ'], userdata)

    def bad_jvsolvefn_y(t, y, yp, v, Jv, userdata):
        y[0] = 0.0
        Jv[:] = userdata['JJ'].dot(v)

    jactimes = CVODEJacTimes(jvsetupfn_hard, bad_jvsolvefn_y)
    solver = CVODE(rhsfn, linsolver='gmres', jactimes=jactimes,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    def bad_jvsolvefn_yp(t, y, yp, v, Jv, userdata):
        yp[0] = 0.0
        Jv[:] = userdata['JJ'].dot(v)

    jactimes = CVODEJacTimes(jvsetupfn_hard, bad_jvsolvefn_yp)
    solver = CVODE(rhsfn, linsolver='gmres', jactimes=jactimes,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

    def bad_jvsolvefn_v(t, y, yp, v, Jv, userdata):
        v[0] = 0.0
        Jv[:] = userdata['JJ'].dot(v)

    jactimes = CVODEJacTimes(jvsetupfn_hard, bad_jvsolvefn_v)
    solver = CVODE(rhsfn, linsolver='gmres', jactimes=jactimes,
                   userdata=userdata)
    with pytest.raises(ValueError, match='read-only'):
        _ = solver.solve(tspan, y0)

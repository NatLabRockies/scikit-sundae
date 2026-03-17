import pickle

import pytest
import numpy as np
import numpy.testing as npt

from sksundae.ida import IDA, IDAResult


def ode(t, y, yp, res):
    res[0] = yp[0] - 0.1
    res[1] = yp[1] - y[1]


def dae(t, y, yp, res):
    res[0] = yp[0] - 0.1
    res[1] = 2*y[0] - y[1]


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


def dae_soln(t, y0):
    if hasattr(t, 'size'):
        y = np.zeros([t.size, 2])
        y[:, 0] = 0.1*t + y0[0]
        y[:, 1] = 2*y[:, 0]
    else:
        y = np.zeros([2])
        y[0] = 0.1*t + y0[0]
        y[1] = 2*y[0]
    return y


def test_ida_ode_solve():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 2])

    solver = IDA(ode, rtol=1e-9, atol=1e-12)

    tspan = np.linspace(0, 10, 11)  # normal solve - user picks times
    soln = solver.solve(tspan, y0, yp0)
    assert len(tspan) > 2 and len(tspan) == len(soln.t)
    npt.assert_allclose(soln.y, ode_soln(soln.t, y0))

    tspan = np.array([0, 10])  # onestep solve - integrator picks times
    soln = solver.solve(tspan, y0, yp0)
    assert len(tspan) == 2 and len(soln.t) > 2
    npt.assert_allclose(soln.y, ode_soln(soln.t, y0))


def test_ida_ode_step():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 2])

    solver = IDA(ode, rtol=1e-9, atol=1e-12)

    with pytest.raises(ValueError):  # have to call init_step first
        _ = solver.step(10)

    soln_0 = solver.init_step(0, y0, yp0)
    npt.assert_allclose(soln_0.y, ode_soln(soln_0.t, y0))

    soln_10 = solver.step(10)
    npt.assert_allclose(soln_10.y, ode_soln(soln_10.t, y0))

    soln_1000 = solver.step(1000, method='onestep', tstop=1000)
    assert soln_1000.t > 10 and soln_1000.t < 1000
    npt.assert_allclose(soln_1000.y, ode_soln(soln_1000.t, y0))


def test_ida_dae_solve():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1])

    tspan = np.linspace(0, 10, 11)  # normal solve - user picks times
    soln = solver.solve(tspan, y0, yp0)
    assert len(tspan) > 2 and len(tspan) == len(soln.t)
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))

    tspan = np.array([0, 10])  # onestep solve - integrator picks times
    soln = solver.solve(tspan, y0, yp0)
    assert len(tspan) == 2 and len(soln.t) > 2
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))


def test_ida_dae_step():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1])

    with pytest.raises(ValueError):  # have to call init_step first
        _ = solver.step(10)

    soln_0 = solver.init_step(0, y0, yp0)
    npt.assert_allclose(soln_0.y, dae_soln(soln_0.t, y0))

    soln_10 = solver.step(10)
    npt.assert_allclose(soln_10.y, dae_soln(soln_10.t, y0))

    soln_1000 = solver.step(1000, method='onestep', tstop=1000)
    assert soln_1000.t > 10 and soln_1000.t < 1000
    npt.assert_allclose(soln_1000.y, dae_soln(soln_1000.t, y0))


def test_ida_userdata():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    def dae_w_data(t, y, yp, res, userdata):
        res[0] = yp[0] - userdata['rate']
        res[1] = userdata['ratio']*y[0] - y[1]

    with pytest.raises(ValueError):  # userdata keyword arg cannot be None
        _ = IDA(dae_w_data, rtol=1e-9, atol=1e-12)

    solver = IDA(dae_w_data, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 userdata={'rate': 0.1, 'ratio': 2})

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))


def test_ida_initcond():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0])

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 calc_initcond='yp0')

    soln = solver.init_step(0, y0, [0, 0])
    npt.assert_allclose(soln.yp, yp0)

    tspan = np.array([0, 10])
    soln = solver.solve(tspan, y0, [0, 0])
    npt.assert_allclose(soln.yp[0], yp0)


def test_ida_atol():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    with pytest.raises(ValueError):  # atol dimension doesn't match y
        solver = IDA(dae, rtol=1e-9, atol=[1e-12]*3, algebraic_idx=[1])
        _ = solver.init_step(0, y0, yp0)

    solver = IDA(dae, rtol=1e-9, atol=[1e-12, 1e-12], algebraic_idx=[1])
    soln_0 = solver.init_step(0, y0, yp0)
    npt.assert_allclose(soln_0.y, dae_soln(soln_0.t, y0))


def test_ida_linsolver():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 2])

    with pytest.raises(ValueError):  # forgot bandwidth(s)
        _ = IDA(ode, rtol=1e-9, atol=1e-12, linsolver='band')

    solver = IDA(ode, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 linsolver='band', lband=0, uband=0)

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)
    npt.assert_allclose(soln.y, ode_soln(soln.t, y0))


@pytest.mark.parametrize('linsolver', ['dense', 'band'])
def test_ida_sparsity(linsolver):  # using idaLSSparseDQJac for dense/band
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    sparsity = np.array([[1, 0], [1, 1]])

    options = {}
    if linsolver == 'band':
        options.update({'lband': 1, 'uband': 0})

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 linsolver=linsolver, sparsity=sparsity, **options)

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))


def test_ida_constraints():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    # cannot satisfy constraints
    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 constraints_idx=[0, 1], constraints_type=[-2, -2])

    _ = solver.init_step(0, y0, yp0)
    soln = solver.step(10)
    assert not soln.success

    # can satisfy constraints
    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 constraints_idx=[0, 1], constraints_type=[2, 2])

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))


def test_ida_eventsfn():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    def eventsfn(t, y, yp, events):
        events[0] = y[0] - 1.55

    with pytest.raises(ValueError):  # forgot num_events
        _ = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                eventsfn=eventsfn)

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 eventsfn=eventsfn, num_events=1)

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)
    assert soln.t[-1] < tspan[-1]  # event was terminal
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))

    npt.assert_allclose(soln.i_events[0], [1])  # event detected correctly
    npt.assert_allclose(soln.y_events[0][0], 1.55)

    npt.assert_allclose(soln.t_events[0], soln.t[-1])  # event was concatenated
    npt.assert_allclose(soln.y_events[0], soln.y[-1])
    npt.assert_allclose(soln.yp_events[0], soln.yp[-1])

    eventsfn.terminal = [False]

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 eventsfn=eventsfn, num_events=1)

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)
    npt.assert_allclose(soln.t[-1], tspan[-1])  # event wasn't terminal
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))

    npt.assert_allclose(soln.i_events[0], [1])  # event detected correctly
    npt.assert_allclose(soln.y_events[0][0], 1.55)

    assert not np.isclose(soln.t_events[0], soln.t[-1])  # didn't concatenate
    assert not np.allclose(soln.y_events[0], soln.y[-1])

    eventsfn.direction = [-1]

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 eventsfn=eventsfn, num_events=1)

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)
    npt.assert_allclose(soln.t[-1], tspan[-1])  # event didn't trigger
    npt.assert_allclose(soln.y, dae_soln(soln.t, y0))

    assert soln.i_events is None  # event didn't trigger
    assert soln.t_events is None
    assert soln.y_events is None
    assert soln.yp_events is None


@pytest.mark.parametrize('linsolver', ['dense', 'band'])
def test_ida_jacfn(linsolver):
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 2])
    tspan = np.linspace(0, 10, 11)

    log = []

    def ode_jacfn(t, y, yp, res, cj, JJ):
        log.append('ode_jacfn called.')
        JJ[0, 0] = cj
        JJ[1, 1] = -1 + cj

    options = {}
    if linsolver == 'band':
        options.update({'lband': 0, 'uband': 0})

    # warn about preference between sparsity and jacfn
    with pytest.warns(UserWarning):
        solver = IDA(ode, jacfn=ode_jacfn, sparsity=np.ones((2, 2)),
                     linsolver=linsolver, **options)

    assert log == []  # empty log to start
    soln = solver.solve(tspan, y0, yp0)
    assert 'ode_jacfn called.' in log  # ode_jacfn was called, not ignored

    npt.assert_allclose(soln.y, ode_soln(soln.t, y0), rtol=1e-4)

    # without sparsity, still calls jacfn (ode)
    solver = IDA(ode, rtol=1e-9, atol=1e-12, jacfn=ode_jacfn,
                 linsolver=linsolver, **options)

    log.clear()

    assert log == []  # empty log to start
    soln = solver.solve(tspan, y0, yp0)
    assert 'ode_jacfn called.' in log  # ode_jacfn was called, not ignored

    npt.assert_allclose(soln.y, ode_soln(soln.t, y0), rtol=1e-4)

    # without sparsity, still calls jacfn (dae)
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])
    tspan = np.linspace(0, 10, 11)

    def dae_jacfn(t, y, yp, res, cj, JJ):
        log.append('dae_jacfn called.')
        JJ[0, 0] = cj
        JJ[1, 0] = 2
        JJ[1, 1] = -1

    options = {}
    if linsolver == 'band':
        options.update({'lband': 1, 'uband': 0})

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 jacfn=dae_jacfn, linsolver=linsolver, **options)

    log.clear()

    assert log == []  # empty log to start
    soln = solver.solve(tspan, y0, yp0)
    assert 'dae_jacfn called.' in log  # dae_jacfn was called, not ignored

    npt.assert_allclose(soln.y, dae_soln(soln.t, y0), rtol=1e-4)


def test_failures_on_exceptions():

    # exception in resfn
    def bad_dae(t, y, yp, res):
        if t > 1:
            raise ValueError("propagated exception")

        res[0] = yp[0] - 0.1
        res[1] = 2*y[0] - y[1]

    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    solver = IDA(bad_dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1])

    tspan = np.linspace(0, 10, 11)
    with pytest.raises(ValueError, match='propagated exception'):
        _ = solver.solve(tspan, y0, yp0)

    # exceptions in eventsfn
    def eventsfn(t, y, yp, events):
        if t > 1:
            raise ValueError("propagated exception")

        events[0] = y[0] - 1.55

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 eventsfn=eventsfn, num_events=1)

    tspan = np.linspace(0, 10, 11)
    with pytest.raises(ValueError, match='propagated exception'):
        _ = solver.solve(tspan, y0, yp0)

    # exceptions in jacfn
    def jacfn(t, y, yp, res, cj, JJ):
        if t > 1:
            raise ValueError("propagated exception")

        JJ[0, 0] = cj
        JJ[1, 0] = 2
        JJ[1, 1] = -1

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1],
                 jacfn=jacfn)

    tspan = np.linspace(0, 10, 11)
    with pytest.raises(ValueError, match='propagated exception'):
        _ = solver.solve(tspan, y0, yp0)


def test_IDAResult():
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1])

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)

    result = IDAResult(**soln.__dict__)
    assert result.message == soln.message
    assert result.success == soln.success
    assert result.status == result.status
    npt.assert_allclose(result.t, soln.t)
    npt.assert_allclose(result.y, soln.y)
    npt.assert_allclose(result.yp, soln.yp)
    assert result.i_events == soln.i_events  # Don't use allclose here b/c
    assert result.t_events == soln.t_events  # all events are None.
    assert result.y_events == soln.y_events
    assert result.yp_events == soln.yp_events
    assert result.nfev == soln.nfev
    assert result.njev == soln.njev


def test_pickling(tmp_path):
    y0 = np.array([1, 2])
    yp0 = np.array([0.1, 0.2])

    solver = IDA(dae, rtol=1e-9, atol=1e-12, algebraic_idx=[1])

    tspan = np.linspace(0, 10, 11)
    soln = solver.solve(tspan, y0, yp0)

    # pickle the solver
    pkl_file = tmp_path.joinpath('ida_solver.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(solver, f)

    # unpickle the solver
    with open(pkl_file, 'rb') as f:
        loaded_solver = pickle.load(f)

    # solve again with the unpickled solver
    loaded_soln = loaded_solver.solve(tspan, y0, yp0)

    npt.assert_allclose(loaded_soln.t, soln.t)
    npt.assert_allclose(loaded_soln.y, soln.y)

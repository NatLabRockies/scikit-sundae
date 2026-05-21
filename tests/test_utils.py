import time

import pytest
import numpy as np
import sksundae as sun


def test_expected_config():
    config = sun._cy_common.config

    assert config['SUNDIALS_INT_TYPE'] == 'int'
    assert config['SUNDIALS_FLOAT_TYPE'] == 'double'
    assert config['SUNDIALS_BLAS_LAPACK_ENABLED'] == 'True'
    assert config['SUNDIALS_SUPERLUMT_ENABLED'] == 'True'
    assert config['SUNDIALS_SUPERLUMT_THREAD_TYPE'] == 'OPENMP'


def test_rich_result():

    result = sun.utils.RichResult()
    assert result._order_keys == []
    assert repr(result) == 'RichResult()'

    class NewResult(sun.utils.RichResult):
        pass

    result = NewResult()
    assert repr(result) == 'NewResult()'

    class OrderedResult(sun.utils.RichResult):
        _order_keys = ['first', 'second',]

    new = NewResult(second=None, first=None)
    ordered = OrderedResult(second=None, first=None)
    assert new.__dict__ == ordered.__dict__
    assert repr(new) != repr(ordered)


def test_format_float_10():
    from sksundae.utils._rich_result import _format_float_10

    assert _format_float_10(np.inf) == '       inf'
    assert _format_float_10(-np.inf) == '      -inf'
    assert _format_float_10(np.nan) == '       nan'

    assert _format_float_10(0.123456789) == ' 1.235e-01'
    assert _format_float_10(1.234567890) == ' 1.235e+00'
    assert _format_float_10(1234.567890) == ' 1.235e+03'


def test_timer():

    # invalid units
    with pytest.raises(ValueError):
        timer = sun.utils.Timer(units='fake')

    # basic
    def f():
        time.sleep(1e-3)
        return 0.

    with sun.utils.Timer('success') as timer:
        _ = f()

    assert timer.name == 'success'
    assert timer.elapsed_time >= 0.
    assert timer._converter['s'](3600.) == 3600.
    assert timer._converter['min'](3600.) == 60.
    assert timer._converter['h'](3600.) == 1.

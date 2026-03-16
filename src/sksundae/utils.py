"""
General-purpose module for shared utilities across the package.

"""

import numpy as np

__all__ = ['RichResult',]


# RichResult and its formatters are modified copies from scipy._lib._util
class RichResult:
    """Output container with pretty printing."""

    _order_keys = []

    def __init__(self, **kwargs):
        """
        This class is a based off the `_RichResult` class in the `scipy`
        library. It combines a series of formatting functions to make the
        printed 'repr' easy to read. Use this class directly by passing in
        any number of keyword arguments, or use it as a base class to have
        custom classes for different result types.

        Inheriting classes should define the class attribute `_order_keys`
        which is a list of strings that defines how output fields are sorted
        when an instance is printed.

        Parameters
        ----------
        **kwargs : dict, optional
            User-specified keyword arguments. Any number of arguments can be
            given as input. The class simply stores the key/value pairs, makes
            them accessible as attributes, and provides pretty printing.

        Examples
        --------
        The example below demonstrates how to define the `_order_keys` class
        attribute for custom sorting. If arguments are not in the list, they
        are placed at the end based on the order they were given. Note that
        `_order_keys` only provides sorting support and that no errors are
        raised if an argument is not present, e.g., `third` below.

        .. code-block:: python

            import sksundae as sun

            class CustomResult(sun.utils.RichResult):
                _order_keys = ['first', 'second', 'third',]

            result = CustomResult(second=None, last=None, first=None)
            print(result)

        `RichResult` can also be used directly, without any custom sorting.
        Arguments will print based on the order they were input. Instances will
        still have a fully formatted 'repr', including formatted arrays.

        .. code-block:: python

            import numpy as np

            from sksundae.utils import RichResult

            t = np.linspace(0, 1, 1000)
            y = np.random.rand(1000, 5)

            y[0] = np.inf
            y[-1] = np.nan

            result = RichResult(message='Example.', status=0, t=t, y=y)
            print(result)

        After initialization, all key/value pairs are accessible as instance
        attributes.

        .. code-block:: python

            from sksundae.utils import RichResult

            result = RichResult(a=10, b=20, c=30)

            print(result.a*(result.b + result.c))

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        order_keys = getattr(self, '_order_keys')

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list, move to end
                return np.inf

        def sorter(d):
            return sorted(d.items(), key=key)

        if self.__dict__.keys():
            return '\n' + _format_dict(self.__dict__, sorter=sorter) + '\n'
        else:
            return self.__class__.__name__ + '()'


def _indenter(s, n=0):
    """Ensures lines after the first are indented by the specified amount."""

    split = s.split('\n')
    return ('\n' + ' '*n).join(split)


def _format_float_10(x):
    """Returns string representation of floats with exactly ten characters."""

    if np.isposinf(x):
        return '       inf'
    elif np.isneginf(x):
        return '      -inf'
    elif np.isnan(x):
        return '       nan'
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)


def _format_dict(d, n=0, mplus=1, sorter=None):
    """Pretty printer for dictionaries."""

    if isinstance(d, dict):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys

        # indent and format each value... + 2 for ': '
        line_items = []
        for k, v in sorter(d):
            formatted_v = _indenter(_format_dict(v, m+n+2, 0, sorter), m+2)
            line_items.append(k.rjust(m) + ': ' + formatted_v)

        output_str = '\n'.join(line_items)

    else:
        with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _format_float_10}):

            output_str = str(d)

    return output_str

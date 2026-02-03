"""
scikit-SUNDAE provides Python bindings to `SUNDIALS`_ integrators. The implicit
differential algebraic (IDA) solver and C-based variable-coefficient ordinary
differential equations (CVODE) solver are both included.

The name SUNDAE combines (SUN)DIALS and DAE, which stands for differential
algebraic equations. Solvers specific to DAE problems are not frequently
available in Python. An ordinary differential equation (ODE) solver is also
included for completeness. ODEs can be categorized as a subset of DAEs (i.e.,
DAEs with no algebraic constraints).

.. _SUNDIALS: https://sundials.readthedocs.io

Accessing the documentation
---------------------------
Documentation is accessible via Python's ``help()`` function which prints
docstrings from a package, module, function, class, etc. You can also access
the documentation by visiting the website, hosted on Read the Docs. The website
includes search functionality and more detailed examples.

Acknowledgements
----------------
scikit-SUNDAE was written by researchers at the **National Renewable Energy
Laboratory (NREL)**, primarily to solve physics-based battery models. Modeling
in Python typically allows for rapid development and makes code more shareable.
However, there was an identified gap in Python's numerical computing ecosystem:
the lack of accessible DAE solvers. While ODE solvers are widely available in
many packages, DAE solvers are not as prevalent.

scikit-SUNDAE started out as a lighter-weight and easy-to-install alternative
to `scikits-odes`_, which similarly provides SUNDIALS bindings, but requires
building from source. The goal was to offer a simpler installation process,
with binary distributions that are consistent across major platforms (PyPI and
conda). While scikit-SUNDAE's API was mostly modeled after scikits-odes, we
want to point out that our codebase was written from scratch and that this is
a separate, independent package. During development we prioritized:

1. Using scipy-like output containers
2. Adopting event function APIs like `scipy.integrate`_, with a few exceptions
3. Maintaining and testing builds using SUNDIALS releases on `conda-forge`_
4. Setting up the package for binary distribution

Since scikit-SUNDAE installations may include pre-built SUNDIALS libraries, a
`bundled license file`_ is included with all distributions. The bundled license
acknowledges not only SUNDIALS, but any optional solvers that are linked and
distributed as well (e.g., SuperLU_MT, OpenBLAS, and LAPACK).

.. _scikits-odes: https://scikits-odes.readthedocs.io
.. _conda-forge: https://anaconda.org/conda-forge/sundials
.. _scipy.integrate: https://docs.scipy.org/doc/scipy/reference/integrate.html
.. _bundled license file: https://github.com/NREL/scikit-sundae/blob/main/ \
                          LICENSES_bundled

"""

from ._cy_common import SUNDIALS_VERSION

from . import ida
from . import utils
from . import cvode
from . import jacband

__all__ = ['ida', 'utils', 'cvode', 'jacband', 'SUNDIALS_VERSION']

__version__ = '1.1.1'

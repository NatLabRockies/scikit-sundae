<!-- <img alt='Logo' style='width: 75%; min-width: 250px; max-width: 500px;'
 src='https://github.com/NatLabRockies/scikit-sundae/blob/main/images/readme_logo.png?raw=true'/> -->

 # scikit-SUNDAE

[![ci](https://github.com/NatLabRockies/scikit-sundae/actions/workflows/ci.yml/badge.svg)](https://github.com/NatLabRockies/scikit-sundae/actions/workflows/ci.yml) &nbsp;
![coverage](https://github.com/NatLabRockies/scikit-sundae/blob/main/images/coverage.svg?raw=true) &nbsp;
[![license](https://img.shields.io/badge/license-BSD--3-blue.svg)](https://github.com/NatLabRockies/scikit-sundae/blob/main/LICENSE) &nbsp;
[![downloads](https://static.pepy.tech/personalized-badge/scikit-sundae?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/scikit-sundae) &nbsp;
[![pypi](https://img.shields.io/pypi/v/scikit-sundae)](https://pypi.org/project/scikit-sundae)

## Summary
scikit-SUNDAE provides Python bindings to [SUNDIALS](https://sundials.readthedocs.io/) integrators. The implicit differential algebraic (IDA) solver and C-based variable-coefficient ordinary differential equations (CVODE) solver are both included.

The name SUNDAE combines (SUN)DIALS and DAE, which stands for differential algebraic equations. Solvers specific to DAE problems are not frequently available in Python. An ordinary differential equation (ODE) solver is also included for completeness. ODEs can be categorized as a subset of DAEs (i.e., DAEs with no algebraic constraints).

## Installation
scikit-SUNDAE is installable via either `pip` or `conda`. To install from [PyPI](https://pypi.org/project/scikit-sundae/) use the following command.

```
pip install scikit-sundae
```

If you prefer using the `conda` package manager, you can install scikit-SUNDAE from the `conda-forge` channel using the command below.

```
conda install -c conda-forge scikit-sundae
```

Both sources contain binary installations. If your combination of operating system and CPU architecture is not supported, please submit an [issue](https://github.com/NatLabRockies/scikit-sundae/issues/) to let us know. If you'd prefer to build from source, please see the [documentation](https://scikit-sundae.readthedocs.io/en/latest/user_guide/installation.html).

## Get Started
You are now ready to start solving. Run one of the following examples to check your installation. Afterward, check out the [documentation](https://scikit-sundae.readthedocs.io/) for a full list of options (including event functions), detailed examples, and more.

```python
# Use the CVODE integrator to solve the Van der Pol equation

from sksundae.cvode import CVODE
import matplotlib.pyplot as plt

def rhsfn(t, y, yp):
    yp[0] = y[1]
    yp[1] = 1000*(1 - y[0]**2)*y[1] - y[0]

solver = CVODE(rhsfn)
soln = solver.solve([0, 3000], [2, 0])

plt.plot(soln.t, soln.y[:, 0])
plt.show()
```

The `CVODE` solver demonstrated above is only capable of solving pure ODEs. The constant parameters and time span used above match an example given by [MATLAB](https://www.mathworks.com/help/matlab/ref/ode15s.html) for easy comparison. If you are trying to solve a DAE, you will want to use the `IDA` solver instead. A minimal DAE example is given below for the Robertson problem. As with the CVODE example, the parameters below are chosen to match an online [MATLAB](https://www.mathworks.com/help/matlab/ref/ode15s.html) example for easy comparison.

```python
# Use the IDA integrator to solve the Robertson problem

from sksundae.ida import IDA
import matplotlib.pyplot as plt

def resfn(t, y, yp, res):
    res[0] = yp[0] + 0.04*y[0] - 1e4*y[1]*y[2]
    res[1] = yp[1] - 0.04*y[0] + 1e4*y[1]*y[2] + 3e7*y[1]**2
    res[2] = y[0] + y[1] + y[2] - 1

solver = IDA(resfn, algebraic_idx=[2], calc_initcond='yp0')
soln = solver.solve([4e-6, 4e6], [1, 0, 0], [0, 0, 0])

soln.y[:, 1] *= 1e4  # scale y1 so it is visible in the figure

plt.plot(soln.t, soln.y)
plt.legend(['y0', 'y1', 'y2'])
plt.show()
```

**Notes:**
* If you are new to Python, check out [Spyder IDE](https://www.spyder-ide.org/). Spyder is a powerful interactive development environment (IDE) that can make programming in Python more approachable to new users.
* Check the [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) documentation from scipy or the [scipy-dae](https://pypi.org/project/scipy-dae/) package repository if you are looking for common examples to test out and compare against. Translating an example from another package can help you learn how to use scikit-SUNDAE before trying to solve more challenging problems.

## Citing this Work
This work was authored by researchers at the National Laboratory of the Rockies (NLR). If you use use this package in your work, please include the following citation:

> Randall, Corey R. "scikit-SUNDAE: Python bindings to SUNDIALS Differential Algebraic Equation solvers [SWR-24-137]." Computer software, Oct. 2024. url: [github.com/NatLabRockies/scikit-sundae](https://github.com/NatLabRockies/scikit-sundae). doi: [10.11578/dc.20241104.3](https://doi.org/10.11578/dc.20241104.3).

For convenience, we also provide the following for your BibTex:

```
@misc{randall2024sksundae,
  author = {Randall, Corey R.},
  title = {{scikit-SUNDAE: Python bindings to SUNDIALS Differential Algebraic Equation solvers [SWR-24-137]}},
  url = {github.com/NatLabRockies/scikit-sundae},
  month = {Oct.},
  year = {2024},
  doi = {10.11578/dc.20241104.3},
}
```

You should also cite SUNDIALS following their recommendations [here](https://computing.llnl.gov/projects/sundials/publications). Lastly, if you use the sparse linear solver, please cite the SuperLU_MT library using their [reference guide](https://portal.nersc.gov/project/sparse/superlu/referencing.html).

## Acknowledgements
scikit-SUNDAE was originally inspired by [scikits-odes](https://scikits-odes.readthedocs.io/) which also offers Python bindings to SUNDIALS. The API for scikit-SUNDAE was mostly adopted from scikits-odes; however, all of our source code is original. If you are comparing the two:

1. **scikits-odes:** includes solvers from daepack, scipy, and SUNDIALS. The package only provides source distributions, so users must configure and compile SUNDIALS on their own; however, this gives users maximum flexibility to compile against SUNDIALS builds with their choice of precision, optional solvers, etc.
2. **scikit-SUNDAE:** only includes SUNDIALS solvers. Provides sparse solvers, more flexible events function capabilities (e.g., direction detection and terminal flags), and scipy-like output not available in scikits-odes. Both binary and source distributions are available; however, we prioritize compatibility with SUNDIALS releases on conda-forge over general user-compiled builds.

Our binary distributions include pre-compiled dynamic SUNDIALS libraries that also reference libraries like SuperLU_MT, OpenBLAS, and LAPACK. These are self-contained and will not affect other, existing installations you may already have. To be in compliance with each library's distribution requirements, all scikit-SUNDAE distributions include a summary of all licenses (see the `LICENSES_bundled` file). Note that we only link against and distribute packages with a BSD-3 license. Some solvers and options, like KLU, have LGPL licenses and are therefore not compatible to implement nor distribute.

## Contributing
If you'd like to contribute to this package, please look through the existing [issues](https://github.com/NatLabRockies/scikit-sundae/issues). If the bug you've caught or the feature you'd like to add isn't already reported, please submit a new issue. You should also read through the [developer guidelines](https://scikit-sundae.readthedocs.io/en/latest/development/) if you plan to work on the issue yourself.

## Disclaimer
This work was authored by the National Laboratory of the Rockies (NLR), operated by Alliance for Energy Innovation, LLC, for the U.S. Department of Energy (DOE). The views expressed in the repository do not necessarily represent the views of the DOE or the U.S. Government.
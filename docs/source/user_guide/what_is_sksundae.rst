What is scikit-SUNDAE?
======================
scikit-SUNDAE is a Python package that provides bindings to `SUNDIALS <https://sundials.readthedocs.io/>`_ CVODE and IDA integrators. These integrators are widely used for solving complex initial value problems, making scikit-SUNDAE a powerful tool for researchers and engineers working in computational science and engineering.

SUNDIALS, or the SUite of Nonlinear and DIfferential/ALgebraic equation Solvers, is a collection of advanced numerical solvers designed for the efficient and robust solution of differential equations. Within the SUNDIALS suite, CVODE and IDA are typically used for the following:

* **CVODE:** Specializes in solving ordinary differential equations (ODEs) that arise in a variety of fields, such as physics, engineering, and biology. It handles both stiff and non-stiff systems of equations, making it adaptable to many real-world applications.
* **IDA:** Solves differential-algebraic equations (DAEs), which often appear in systems where some of the equations describe algebraic constraints. These problems arise in applications like multi-body dynamics, chemical kinetics, and electrical circuits.

Use cases
=========
Both **CVODE** and **IDA** are ideal for problems that involve systems of coupled differential equations. They are particularly suited to:

* Time-dependent simulations of physical systems
* Kinetics and chemical reaction modeling
* Structural dynamics involving differential-algebraic systems
* Battery modeling, fuel cell simulations, and energy storage applications

Bindings limitations
====================
While scikit-SUNDAE brings the powerful functionality of CVODE and IDA into the Python ecosystem, it does not wrap every feature available in the SUNDIALS library. In particular:

* The package focuses on providing access to the SUNDIALS serial vector interface and default linear/nonlinear solvers. Where appropriate, optional solvers licensed under BSD-3 may also be incorporated (e.g., SuperLU_MT, OpenBLAS, and LAPACK).
* There are no plans to wrap advanced parallel vector interfaces (e.g., MPI, PThreads, CUDA, etc.) nor optional solvers that have licensing conflicts with BSD-3 (e.g., KLU).
* We prioritize generalized features. For example, since IDA only supports left preconditioning we do not have plans to implement the SPFGMR linear solver, which only supports right preconditioning.
* There are no plans to add an interface for adjoint sensitivity analysis. Solving the adjoint problem requires twice as many options and user-defined functions (building the forward and backward problems), which would double the size of this package.

While scikit-SUNDAE offers a convenient interface to key solvers in the SUNDIALS suite, there is no intention to wrap the entire SUNDIALS library. Given the large number of optional solvers and interfaces available (shown in the figure below), wrapping everything would be a significant challenge. Instead, scikit-SUNDAE focuses on providing efficient and streamlined access to the essential features of the CVODE and IDA integrators, ensuring the package remains lightweight and user-friendly.

.. figure:: figures/SUNDIALS_web.png
   :alt: Diagram of SUNDIALS suite.
   :align: center

Acknowledgements
================
We extend our appreciation to the developers and maintainers of the SUNDIALS project for their exceptional work in creating a robust, reliable, and open-source suite of solvers. Full details on the SUNDIALS license and copyright information can be found `here <https://github.com/LLNL/sundials/blob/main/LICENSE>`_. For the sparse solver implementation we would also like to thank the developers of `SuperLU_MT <https://github.com/xiaoyeli/superlu_mt>`_.

We also acknowledge the `scikits-odes <https://scikits-odes.readthedocs.io/>`_ package, which similarly provides Python bindings to SUNDIALS. While scikit-SUNDAE's API was largely modeled after scikits-odes to maintain a familiar structure, it is important to note that scikit-SUNDAE is an independently developed package, sharing no source code with scikits-odes.

If you are comparing scikits-odes to scikit-SUNDAE, you should consider the following differences:

* **scikits-odes:** includes solvers from daepack, scipy, and SUNDIALS. The package only provides source distributions, so users must configure and compile SUNDAILS on their own; however, this gives users maximum flexibility to compile against SUNDIALS builds with their choice of precision, optional solvers, etc.
* **scikit-SUNDAE:** only includes SUNDIALS solvers. Provides sparse solvers, more flexible events function capabilities (e.g., direction detection and terminal flags), and scipy-like output not available in scikits-odes. Both binary and source distributions are available; however, we prioritize compatibility with SUNDIALS releases on conda-forge over general user-compiled builds.

Our binary distributions include pre-compiled dynamic SUNDIALS libraries that also reference libraries like SuperLU_MT, OpenBLAS, and LAPACK. These are self-contained and will not affect other, existing installations you may already have. To be in compliance with each library's distribution requirements, all scikit-SUNDAE distributions include a summary of all licenses (see the `LICENSES_bundled`_ file). Note that we only link against and distribute packages with BSD-3 license. Some solvers and options, like KLU, have LGPL licenses and therefore will not be compatible to implement/distribute.

.. _LICENSES_bundled: https://github.com/NatLabRockies/scikit-sundae/blob/main/LICENSES_bundled

Disclaimer
==========
scikit-SUNDAE was authored by the National Laboratory of the Rockies (NLR), operated by Alliance for Energy Innovation, LLC, for the U.S. Department of Energy (DOE). The views expressed in the project and documentation do not necessarily represent the views of the DOE or the U.S. Government.

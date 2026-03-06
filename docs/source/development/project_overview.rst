Project Overview
================

Introduction
------------
scikit-SUNDAE is a Python package that offers a robust interface to the SUNDIALS CVODE and IDA integrators, specializing in solving initial value problems (IVPs) for both ordinary differential equations (ODEs) and differential algebraic equations (DAEs). Built with performance and ease of use in mind, scikit-SUNDAE simplifies high-performance numerical integration by exposing these powerful solvers. Whether users need to solve stiff or non-stiff ODEs or more complex DAE systems, scikit-SUNDAE offers a streamlined, intuitive interface that balances practicality with technical depth.

The backend is powered by the highly regarded `SUNDIALS <https://sundials.readthedocs.io>`_ C library, but scikit-SUNDAE eliminates the need for users to write any C code - allowing them to focus purely on modeling their systems in Python.

Key Features
^^^^^^^^^^^^
* **Class-like Interface:** scikit-SUNDAE provides Python classes for both the CVODE (for ODEs) and IDA (for DAEs) solvers. Users can create instances of these solvers based on the specific problem they are working on.
* **Flexible Time Integration:** Both solvers support step-by-step integration or integration over a full time interval, giving users control over how they approach their problems.
* **Extensive Solver Options:** The package includes options for tracking event functions, passing explicit Jacobian functions (to avoid numerical approximations), and fine-tuning solver parameters for optimal performance.
* **C-Powered, Python-Focused:** Built on the high-performance SUNDIALS C library, scikit-SUNDAE leverages C's speed while remaining fully accessible to Python developers.
* **No C Knowledge Required:** Users only need to define their ODE or DAE as Python functions, which are then passed to the solver. This makes the package approachable, even for those without C programming experience.

Use Cases
---------
scikit-SUNDAE is ideally suited for solving complex initial value problems, including:

* **Modeling physical systems:** Simulate the dynamics of physical systems, such as mechanical, electrical, or chemical processes, where differential equations naturally arise.
* **Scientific computing and engineering:** Researchers and engineers can use the package to solve ODEs and DAEs encountered in areas like chemical kinetics, fluid dynamics, heat transfer, and control systems.
* **Multi-scale and multi-physics simulations:** With its ability to efficiently handle stiff systems and DAEs, scikit-sundae is well-suited for simulations that involve processes at multiple scales or involve complex interrelated physics.

Target Audience
---------------
scikit-SUNDAE is designed for:

* Researchers in scientific fields such as physics, chemistry, biology, and economics.
* Engineers across industries who deal with dynamic systems and need accurate time integration.
* Mathematicians and computer scientists working on differential equations and dynamical systems.

Anyone in need of a reliable tool to solve ODEs or DAEs, whether for academic research, industrial applications, or educational purposes.

Technology Stack
----------------
scikit-SUNDAE is built using:

* **Cython:** Used to wrap the C-based SUNDIALS library and provide Python bindings.
* **Numpy:** Integrated seamlessly to allow users to work with Numpy arrays as inputs, initial values, and other parameters.
* **SUNDIALS C Library:** The core backend of the solvers, providing highly efficient and mature algorithms for ODEs and DAEs.

Installation is made simple via binary distributions, and dependencies are automatically managed. Multiple versions of Python are supported.

Project Origins
---------------
scikit-SUNDAE was written by researchers at the **National Laboratory of the Rockies (NLR)** primarily to solve physics-based battery models. Modeling in Python typically allows for rapid development and makes codebases more shareable. However, there was an identified gap in Python's numerical computing ecosystem: the lack of accessible DAE solvers. While ODE solvers are widely available in many packages, DAE solvers are not as prevalent.

scikit-SUNDAE started out as a replacement for `scikits-odes <https://scikits-odes.readthedocs.io>`_, which also provides SUNDIALS bindings, but requires building from source. The goal was to offer a simpler installation process, with binary distributions that are consistent across major platforms (PyPI and conda).

Roadmap and Future Directions
-----------------------------
While scikit-SUNDAE is currently feature-complete with regard to its core solvers (CVODE and IDA), the project will continue to evolve in alignment with new releases of Python and SUNDIALS. As new features are added to the SUNDIALS library, the package will be updated to maintain compatibility and take advantage of performance improvements.

While there are no immediate plans to add new solvers or additional features, scikit-SUNDAE remains open to feature requests. The development team will evaluate suggestions on a case-by-case basis.

Contributions
-------------
scikit-SUNDAE is fully functional and already includes most of the planned features. However, we welcome contributions and ideas for new solvers or additional options. If you're interested in contributing, please submit an `issue <https://github.com/NatLabRockies/scikit-sundae/issues>`_ to start a discussion about your proposed feature.

Whether you want to fix bugs, write tests, or suggest new functionality, contributions are welcome! Any major changes should be preceded by an open discussion, ensuring alignment with the project's goals and structure.

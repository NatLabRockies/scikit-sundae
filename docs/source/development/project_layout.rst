Project Layout
==============
scikit-SUNDAE is organized to provide clarity and structure, making it easy for developers to navigate and contribute. Below is an outline of the key directories and files, along with guidelines for working within them.

Root Directory
--------------
The root directory contains the most important files and folders necessary for development:

* **src/:** The core package code resides in this directory. This is the primary folder developers will interact with when modifying or adding features.
* **pyproject.toml:** This file contains the project's build system configurations and dependencies. If you need to add or modify dependencies, you should do so in this file.
* **setup.py:** Specifies how the Cython extensions get built, including where to look for users' SUNDIALS installations that are linked against.
* **noxfile.py:** Contains automation scripts for tasks like testing, linting, formatting, and building documentation. Developers should use `nox` sessions as needed to ensure code quality and consistency.
* **tests/:** This is where all unit tests and integration tests are stored. Bug fixes and new feastures should always be paired with new tests.
* **docs/:** Contains documentation files for the project. Developers contributing to the documentation should work here, particularly if adding or improving developer guides or API references.

Source Directory
----------------
The `src/` directory contains the main package code. Using this structure ensures that local imports during development come from the installed package rather than accidental imports from the source files themselves.

Organization
^^^^^^^^^^^^
While there are multiple files in the top-level package, they all only contribute to three "small" submodules. As the package grows, the organization may need to evolve to include subfolders, however, at present the package is manageable without subfolders. If you are not familiar with Cython, you will likely want to read more about it and even try making a practice package before you dive into the scikit-SUNDAE source code.

File names follow a convention based on their purpose:

* **Start with `c_`:** These are header-like `.pxd` files that mimic the SUNDIALS C library. These control what functions from the C headers get exposed to Python.
* **Starts with `_cy`:** May come in pairs (`.pxd` and `.pyx`). The `.pyx` files contain the "true" source code. This is where the imported C functions are used and wrapped. The `.pxd` files are header-like so that `cimport` can be used to import the functions from the corresponding `.pyx` file into other `.pyx` files.
* **Python files:** The regular Python files (`.py` extensions) have little-to-no actual functionality. They primarily exist to add sphinx-compatible documentation since we have yet to find a good way to link sphinx to the Cython files directly.

Submodules
^^^^^^^^^^
There are three submodules that handle specific functionality:

* `common`: Contains functions and/or classes that are useful to all solvers. For example, a wrapper class for solutions. Also includes an attribute to store `SUNDIALS_VERSION`.
* `cvode`: Holds the CVODE solver class and its results wrapper. The CVODE class is recommended for all ODE problems, even though IDA can also solve pure ODEs.
* `ida`: Includes both the IDA solver class and its results wrapper. The IDA class is required for DAE problems since CVODE cannot support the algebraic constraints.

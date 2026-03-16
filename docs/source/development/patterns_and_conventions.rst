Patterns and Conventions
========================

File Organization
-----------------
It is preferred to have more files with fewer lines of code rather than fewer, larger files. This keeps the codebase easier to navigate and review. As a rule of thumb, classes and functions that are long should be in their own file, while shorter, related items can be grouped together. However, take care to not group unrelated classes and/or functions just because they are short. It is okay for these to still be in their own files if they are unique and cannot be categorized to fit in with other classes/functions.

To maintain ease of access for users, all user-facing functions and classes should be no more than three levels deep from the top-level of the package. This ensures that users do not need to navigate through excessive subpackages or submodules to find the tools they need. Keeping interfaces easily discoverable improves usability and reduces friction when working with the package. With this in mind, it is still okay for developers to have nested code, however, they should import user-facing functionality into the package or some subpackage that makes it more accessible.

Naming Convention
-----------------
To ensure consistency and ease of development, the following conventions are enforced:

1. File names: 
    Depending on the contents and purpose of the file, it should be named as follows:

    - Start with `c_` if its purpose is to link to C headers from the SUNDIALS library. The rest of the name should be consistent with SUNDIALS headers folders.
    - Start with `_cy` if the file is written in Cython. If both a `.pxd` and `.pyx` file are needed, they should share the same name.
    - Regular Python files should be short and in camel case. These will correspond to the submodule names that the user sees. At present, these include very little actual code and are primarily used to add sphinx-compatible documentation. 

2. Class and function names: 
    Classes use `CamelCase`, while functions and methods use `snake_case`. Classes should generally be in their own file unless grouped logically with other helper functions or classes. When a part of a classname is an abbreviation, it should all be capitalized, e.g., `IDA` for the implicit differential algebraic solver.

Import Considerations
---------------------

Ordering
^^^^^^^^
In our codebase, import statements are organized into three distinct groups based on where the modules originate. This helps keep imports clean and maintainable. The groups, in order, are:

1. **Standard Library Imports:** These come from Python's built-in standard library.
2. **Dependency Imports:** Imports from external dependencies installed via package managers (e.g., `pip` or `conda`). When necessary, this also includes `cimport` dependencies, which should be placed near the regular `import` dependency if both are needed.
3. **Local Package Imports:** Imports that come from within our package. When it makes sense, it is okay to split these local imports into their own subsections. One example of this would be to separate out Cython vs. Python imports. Another would be to separate out header-like imports from the SUNDIALS library from developer-written Cython functions.

Within each group, we generally list imports in ascending order of their length (shortest to longest), as shown in the example below. This helps maintain a neat and consistent style throughout the code. Note that it is not necessary to comment each grouped section, this is only done for clarity in the example.

.. code-block:: cython

    # Standard Library Imports
    import os
    import sys
    import datetime

    # Dependency Imports
    import numpy as np
    cimport numpy as np

    # Local Package Imports - C headers
    from .c_sundials cimport *
    
    # Local Package Imports - local cython 
    from ._cy_common cimport np2svec, svec2np, np2smat

    # Local Package Imports - local python
    from .common import RichResult

Note in the example above that we use the wildcard `*` import to get all of the functions from the `c_sundials` file. This is generally not great practice and should be avoided when very few functions are needed, as demonstrated in the other local imports. However, we do allow this as needed, primarily when importing from `.pxd` files using `cimport`.

Placement
^^^^^^^^^
* Common dependencies that are used across multiple functions should be imported at the top of the module.
* For heavier dependencies or rarely used ones, consider importing them only where needed (within functions/methods) to minimize unnecessary load times.
* Regardless of placement, at the top of a file of within a function/method, ordering within each group should follow the ordering listed above.

Class Considerations
--------------------
For class definitions, we follow a specific ordering convention to make it easier to navigate through the code:

1. **Magic Methods:** These special methods (e.g., `__init__`, `__repr__`, etc.) come first. They define key behaviors of the class. There is one exception here, `__dealloc__`, which should always be the very last method.
2. **Cython and Hidden Methods:** Cython methods use `cdef` and their names should start with a leading underscore `_`, whether they are planned to be exposed or not. Regular hidden methods should also be grouped here, also starting with an underscore.
3. **User-Facing Methods:** These are the public methods intended for external use. They define the class's core functionality for users. In some cases, these may just wrap a `cdef` method by the same name.

In some cases, exceptions to this order may be made, particularly if moving a hidden method closer to a user-facing method improves readability. However, this should be done with discretion and only when it helps clarify the flow of the class's logic. See below for an example.

.. code-block:: cython

    class MyClass:
        # Magic Methods
        def __cinit__(self, value):
            self.value = value
        
        def __repr__(self):
            return f"MyClass(value={self.value})"

        # Cython and Hidden Methods
        cdef _do_something(self):
            pass

        def _helper_func(self): 
            pass
        
        # User-Facing Methods
        def do_something(self):
            self._helper_function()
            return self._do_something()  # wraps _do_something, but calls
                                         # _helper_func first to preprocess
        # Special __dealloc__ method
        cdef __dealloc__(self):
            pass        

Development Tools
-----------------
For ease of development, tools and dependencies for linting, formatting, spellchecking, testing, and documentation building are included as optional dependencies. Installing these is as simple as running the following::

    pip install -e .[dev]

In addition, developers should use `nox` to automate many tasks:

* `nox -s tests` - run tests with coverage reports
* `nox -s linter` - lint and format the code
* `nox -s codespell` - check for and fix misspellings
* `nox -s pre-commit` - run pre-commit checks (all above)
* `nox -s docs` - build the documentation
* `nox -s rebuild` - rebuild the Cython extensions in place

Use these tools to ensure the code remains clean and follows best practices.

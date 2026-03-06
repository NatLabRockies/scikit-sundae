Code Style and Linting
======================
Maintaining a consistent code style and adhering to linting rules is crucial for ensuring code quality and readability. This page outlines the guidelines and tools used for code style and linting in our project.

Styling Guidelines
------------------
We adhere to `PEP8 <https://peps.python.org/pep-0008/>`_ with minimal exceptions. Minor adjustments to spacing (around operators, under/over-indentation) are allowed when they improve clarity. We include a `.flake8` configuration file in `.github/linters` that specifies these exceptions. Developers should configure their IDEs with this file to ensure consistency.

Code Formatting
---------------
While `black <https://black.readthedocs.io/en/stable/>`_ is a popular auto-formatting package, we do not permit it to be used for this codebase. Although it adheres to the same PEP8 standards that we follow, the `black` styling can be a bit more opinionated at times and does not always help improve clarity. For those still looking for auto-formatting, we permit the use of `autopep8 <https://github.com/hhatto/autopep8>`_ when paired with the `.flake8` configuration file found in `.github/linters`. IDEs supporting `autopep8` should be configured accordingly. Developers can also run the formatter manually using::

    nox -s linter -- format 

When used with the optional `format` argument, this `nox` command will first run the auto-formatter and then check for errors. This means that if errors persist, then `autopep8` was unable to address them and that they must be addressed manually.

Enforcement
-----------
Style and linting are enforced through Continuous Integration (CI). Developers should perform local checks using::

    nox -s linter 

For a comprehensive suite of checks, including unit tests and spelling in comments, run::

    nox -s pre-commit

This will ensure that all code meets the required standards before pushing changes. If you skip local checks, the CI will catch issues during the push process. Failed tests may result in a delayed reviewer assignments when you open pull requests.

Documentation
-------------
Python files should be documented using the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ docstring format. All classes, methods, and functions must have clear docstrings, including hidden methods/functions. Use type hints to specify input and output types. Code should be readable with minimal inline comments, though particularly complex sections should include additional explanations.

Cython files do not need to adhere to the same docstring standards as the Python files. The Python file docstrings are scrapped to build the API documentation using sphinx, and are also what gets printed to the terminal when using `help()`. However, since the Cython files actually contain all of the complex logic, they should still be readable and well commented. Prioritize making the code read easily without comments, but include inline comments as needed. It is more approapriate to have longer and more frequent inline commenting in the Cython files compared to the Python files.

Additional Preferences
----------------------
When it comes to string quotation, we have a few specific preferences to maintain consistency across the codebase:

* **Single quotes:** 

    Use single quotes (`'`) for string variables, dictionary keys, and other standard strings. For example:

    .. code-block:: python

        my_string = 'This is a string.'
        my_dict = {'key': 'value'}

* **Double quotes:** 

    Use double quotes (`"`) for strings that are part of exception messages, print statements, or special string types such as formatted or raw strings. For example:

    .. code-block:: python

        print("This is a print statement.")
        raise ValueError("This is an exception message.")
        formatted_string = f"This is a formatted string: {value}"
        raw_string = r"This is a raw string."

    In addition, double quotes should EXCLUSIVELY be used in all instances for Cython files. This better mimics the C language, and since Cython is a mixture of Python and C, we have a strong preference to only use one type of quote in these files.

By following these conventions, we aim to enhance readability and maintain consistency in how strings are handled throughout the codebase.

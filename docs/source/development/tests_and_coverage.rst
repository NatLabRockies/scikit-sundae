Tests and Coverage
==================

Overview
--------
Testing and coverage are critical to maintaining code quality and ensuring that our software behaves as expected. This page outlines our practices for writing and running tests, measuring coverage, and maintaining high standards in our codebase.

Testing Practices
-----------------

* Test organization
    Tests should be organized first by module and then by class and/or function. This helps in managing and locating tests effectively. Avoid grouping tests into a single file just because they share similar functions. Instead, organize them based on their associated modules.

* Naming conventions
    All test folders, files, and functions should start with `test_` followed by a descriptive name (using snake case) indicating what is being tested. For example, `test_calculate_total_price` is preferable to `test_Calculator` unless the class is simple enough to be covered with a single test. Design tests to cover specific units, features, or applications of the class or function. 

* Test data
    Use fixtures where appropriate. If mock data is necessary, make a subfolder in the `tests/` to store it in. Make sure the file(s) have descriptive names. Ensure that test data is manageable and not overly complex. We include a `C_programs` folder than contains a couple examples written in C and run with SUNDIALS directly to compare against. Feel free to add more examples here as appropriate.

Running Tests
-------------
The full test suite can be run locally using::

    nox -s tests 

Alternatively, you can run tests from a specific file using::

    pytest tests/test_file.py 

where `test_file.py` is the file that includes the tests you'd like to run. Generally, you will want to run individual files when you are iterating back and forth between fixing bugs, adding features, and writing new tests. However, you should always run the full test suite once you are finished and prior to any commits and/or pushes to your repository.

If you forget to run tests locally they will still be run as part of the continuous integration (CI) workflow on your next push or pull request. While we only expect you to run your tests locally using the most recent stable version of Python, the CI workflow will also run the full test suite using older versions of Python and will check that tests pass on all major operating systems. 

Failing Tests
^^^^^^^^^^^^^
It is possible that although your local tests work that one of the older versions of Python, or even the newest version of Python on a different machine, may fail. In these cases, you should check the GitHub actions logs and address the issue. Ask for help from other developers using the `Discussions <https://github.com/NatLabRockies/scikit-sundae/discussions>`_ page if you ever feel stuck.

A good place to start if your tests are only failing on older Python versions is to setup a second, temporary, development environment with one of the older Python versions. All `nox` commands will still function the same way. This can help you run the failing tests locally instead of continuously pushing to GitHub. After failed tests are resolved, make sure you move back to using your primary development environment for future work.

Coverage 
--------
We use `pytest` along with the `pytest-cov` extension to measure code coverage. The configuration and reports are automatically set and generated for you when you use::

    nox -s tests 

After tests finish running, you can check the coverage by opening the `index.html` file in the `reports/htmlcov/` folder. This will help you navigate through the source code files to see which lines are missing coverage.

Excluding Lines
^^^^^^^^^^^^^^^
In some cases there will be lines of code that do not need to be covered. For example, `if TYPE_CHECKING` line of code does not get run during testing and will therefore never be "covered". In this case and a few others, it is okay to use the directive `# pragma: no cover` to ignore a line (or section) or code. For example,

.. code-block:: python 

    from tying import TYPE_CHECKING

    if TYPE_CHECKING:  # pragma: no cover 
        from numpy import ndarray
        from pandas import DataFrame

We strive to achieve 100% test coverage (excluding lines marked by `# pragma: no cover`); however, do not use this to avoid writing tests for challenging code. Comprehensive testing is essential for maintaining project success.

Performance Testing
-------------------
Tests should prioritize functionality. We do not write performance tests into the test suite. If you are optimizing performance, include examples in your pull request to compare the current and new implementations. Once it is confirmed that the new implementation is an improvement over the old, performance tests can be removed.

# scikit-SUNDAE Changelog

## [Unreleased](https://github.com/NatLabRockies/scikit-sundae)

### New Features
- Move to newest SUNDIALS v7.6 for CI builds/tests ([#44](https://github.com/NatLabRockies/scikit-sundae/pull/44))
- Custom `__reduce__` methods, allowing solvers to be serialized ([#38](https://github.com/NatLabRockies/scikit-sundae/pull/38))

### Optimizations

### Bug Fixes
- Ensures exception propagations work correctly with numpy 2.4 release ([#41](https://github.com/NatLabRockies/scikit-sundae/pull/41))

### Breaking Changes

### Chores
- Make GitHub hyperlinks reference new org name `NREL` -> `NatLabRockies` ([#42](https://github.com/NatLabRockies/scikit-sundae/pull/42))
- Allow single backticks for sphinx inline code (`default_role = 'literal'`) ([#40](https://github.com/NatLabRockies/scikit-sundae/pull/40))
- Rebrand NREL to NLR, and include name change for Alliance as well ([#39](https://github.com/NatLabRockies/scikit-sundae/pull/39))

## [v1.1.0](https://github.com/NatLabRockies/scikit-sundae/tree/v1.1.0)

### New Features
- Enforce constraints when `**LSSparseDQJac` funcs are used ([#32](https://github.com/NatLabRockies/scikit-sundae/pull/32))
- Drop Python 3.9 and add support for 3.14 in tests/release ([#31](https://github.com/NatLabRockies/scikit-sundae/pull/31))
- Move to newest SUNDIALS v7.5 for CI builds/tests ([#30](https://github.com/NatLabRockies/scikit-sundae/pull/30))
- Add version warning banner to docs for dev and older releases ([#28](https://github.com/NatLabRockies/scikit-sundae/pull/28]))
- Move to newest SUNDIALS v7.4 for CI builds/tests ([#23](https://github.com/NatLabRockies/scikit-sundae/pull/23))
- Move to newest SUNDIALS v7.3 for CI builds/tests ([#16](https://github.com/NatLabRockies/scikit-sundae/pull/16))
- Add `reduce_bandwidth` function to help restructure sparse problems ([#15](https://github.com/NatLabRockies/scikit-sundae/pull/15))
- Implement interfaces for Jacobian-vector products ([#13](https://github.com/NatLabRockies/scikit-sundae/pull/13))
- Allow preconditioning for iterative solvers ([#12](https://github.com/NatLabRockies/scikit-sundae/pull/12))
- Enable OpenBLAS-linked LAPACK linear solvers ([#11](https://github.com/NatLabRockies/scikit-sundae/pull/11))
- Add iterative linear solvers to IDA and CVODE ([#10](https://github.com/NatLabRockies/scikit-sundae/pull/10))
- Allow `numpy` types in options checks for both `CVODE` and `IDA` ([#8](https://github.com/NatLabRockies/scikit-sundae/pull/8))
- Expose linear solver option that uses SuperLU_MT ([#6](https://github.com/NatLabRockies/scikit-sundae/pull/6))
- New `jacband` module for support finding sparsity/bandwidth ([#6](https://github.com/NatLabRockies/scikit-sundae/pull/6))
- Custom `sparseDQJac` routines available by supplying `sparsity` ([#6](https://github.com/NatLabRockies/scikit-sundae/pull/6))
- Changed signature inspections to support decorated `jit` functions ([#3](https://github.com/NatLabRockies/scikit-sundae/pull/3))

### Optimizations
- Use `np.testing` where possible in tests for more informative fail statements ([#14](https://github.com/NatLabRockies/scikit-sundae/pull/14))
- Updates to be compliant with Cython deprecations of `IF/ELIF/ELSE` and `DEF` ([#5](https://github.com/NatLabRockies/scikit-sundae/pull/5))
- Use single-line memory views and pointer addressing for `np2ptr` and `np2smat` ([#5](https://github.com/NatLabRockies/scikit-sundae/pull/5))
- Use `micromamba` instead of `miniconda` in CI ([#3](https://github.com/NatLabRockies/scikit-sundae/pull/3))

### Bug Fixes
- Move to `cibuildwheel` for releases due to segfaults on Linux ([#33](https://github.com/NatLabRockies/scikit-sundae/pull/33))
- Fixed import typo in docstring examples for `RichResult` ([#29](https://github.com/NatLabRockies/scikit-sundae/pull/29))
- Resolve exception propagation consistently for Cython v3.1 and up ([#20](https://github.com/NatLabRockies/scikit-sundae/pull/20))
- Fix memory leak when `init_step` is repeatedly called in `CVODE` and `IDA` ([#19](https://github.com/NatLabRockies/scikit-sundae/pull/19))
- Add `sign_y` terms and default to `np.float64` for floating type in `j_pattern` ([#7](https://github.com/NatLabRockies/scikit-sundae/pull/7))

### Breaking Changes
None.

## [v1.0.0](https://github.com/NatLabRockies/scikit-sundae/tree/v1.0.0)
This is the first official release of scikit-SUNDAE. Main features/capabilities are listed below.

### Features
- Python errors can be propagated through Cython wrappers
- Implicit differential algebraic (IDA) solver for differential algebraic equations (DAEs)
- C-based variable-coeffecients ordinary differential equations (CVODE) solver
- Events functions with scipy-like API, including "terminal" and "direction" options
- Dense and banded linear solver options in both IDA and CVODE
- Option for user-supplied Jacobian function in both IDA and CVODE
- scipy-like `RichResult` output containers

### Notes
- Tests check solutions against C programs
- Source/binary distributions available on [PyPI](https://pypi.org/project/scikit-sundae)
- Documentation available on [Read the Docs](https://scikit-sundae.readthedocs.io/)

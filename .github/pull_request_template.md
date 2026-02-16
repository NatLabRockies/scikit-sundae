# Description
Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

Fixes # (issue)

## Type of change
Please add a line in the relevant section of [CHANGELOG.md](https://github.com/NatLabRockies/scikit-sundae/blob/main/CHANGELOG.md) to document the change (include PR #) - note reverse order of PR #s. If necessary, also add to the list of breaking changes.

- [ ] New feature (non-breaking change which adds functionality)
- [ ] Optimization (back-end change that improves speed/readability/etc.)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] Chore (any other clean up, maintenance, etc.)

# Key checklist
- [ ] No style issues: `$ nox -s linter [-- format]`
- [ ] Code is free of misspellings: `$ nox -s codespell [-- write]`
- [ ] All tests pass: `$ nox -s tests`
- [ ] Badges are updated: `$ nox -s badges`

The optional `-- format` and `-- write` arguments (see above) attempt to correct formatting issues prior to running the linter, and spelling mistakes prior to running the spellcheck, respectively. You can also run all of the above checks using `$ nox -s pre-commit` instead of running them individually.

## Further checks:
- [ ] The documentation builds: `$ nox -s docs`.
- [ ] Code is commented, particularly in hard-to-understand areas.
- [ ] Tests are added that prove fix is effective or that feature works.
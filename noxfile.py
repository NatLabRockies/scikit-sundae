# noxfile.py

import os
import shutil

import nox

nox.options.sessions = []


@nox.session(name='cleanup', python=False)
def run_cleanup(_) -> None:
    """Use os/shutil to remove some files/directories"""
    if os.path.exists('.coverage'):
        os.remove('.coverage')

    folders = ['.pytest_cache', '__pycache__']
    for f in folders:
        if os.path.exists(f):
            shutil.rmtree(f)


@nox.session(name='linter', python=False)
def run_ruff(session: nox.Session) -> None:
    """
    Run ruff to check for linting errors.

    Use the optional 'format' or 'format-unsafe' arguments to run ruff with the
    --fix or --unsafe-fixes option prior to the linter. You can also use 'stats'
    to show a summary of the found errors rather than a full report.

    """
    session.run('pip', 'install', '--upgrade', '--quiet', 'ruff')

    command = ['ruff', 'check']
    if 'stats' in session.posargs:
        command.append('--statistics')

    if 'format' in session.posargs:
        command.append('--fix')
    elif 'format-unsafe' in session.posargs:
        command.extend(['--fix', '--unsafe-fixes'])

    session.run(*command)


@nox.session(name='codespell', python=False)
def run_codespell(session: nox.Session) -> None:
    """
    Run codespell with the github config file

    Use the optional 'write' argument to write the corrections directly into
    the files. Otherwise, you will only see a summary of the found errors.

    """
    session.run('pip', 'install', '--upgrade', '--quiet', 'codespell')

    command = ['codespell', '--config=.github/linters/.codespellrc']
    if 'write' in session.posargs:
        command.insert(1, '-w')

    session.run(*command)


@nox.session(name='spellcheck', python=False)
def run_spellcheck(session: nox.Session) -> None:
    """
    Run codespell with docs files included

    Use the optional 'write' argument to write the corrections directly into
    the files. Otherwise, you will only see a summary of the found errors.

    """
    run_codespell(session)

    command = ['codespell', '--config=.github/linters/.codespellrc']
    if 'write' in session.posargs:
        command.insert(1, '-w')

    session.run(*command, 'docs/source')


@nox.session(name='tests', python=False)
def run_pytest(session: nox.Session) -> None:
    """
    Run pytest and generate test/coverage reports

    Use the optional 'parallel' argument to run the tests in parallel. As just
    a flag, the number of workers will be determined automatically. Otherwise,
    you can specify the number of workers using an int, e.g., parallel=4.

    """
    if 'no-reports' in session.posargs:
        command = [
            'pytest',
            '--cov=sksundae',
            'tests/',
        ]
    elif 'codecov' in session.posargs:
        command = [
            'pytest',
            '--cov=sksundae',
            '--cov-report=xml',
            'tests/',
        ]
    else:
        os.makedirs('reports', exist_ok=True)

        command = [
            'pytest',
            '--cov=sksundae',
            '--cov-report=html:reports/htmlcov',
            '--cov-report=xml:reports/coverage.xml',
            '--junitxml=reports/junit.xml',
            '--html=reports/pytest.html',
            '--self-contained-html',
            'tests/',
        ]

    for arg in session.posargs:
        if arg.startswith('parallel='):
            command[1:1] = ['-n', arg.split('=')[-1]]
        elif arg.startswith('parallel'):
            command[1:1] = ['-n', 'auto']

    os.environ['MPLBACKEND'] = 'Agg'
    session.run(*command)


@nox.session(name='badges', python=False)
def run_genbadge(session: nox.Session) -> None:
    """Run genbadge to make test/coverage badges"""
    session.run(
        'genbadge', 'coverage', '-l',
        '-i', 'reports/coverage.xml',
        '-o', 'images/coverage.svg',
    )

    session.run(
        'genbadge', 'tests', '-l',
        '-i', 'reports/junit.xml',
        '-o', 'images/tests.svg',
    )


@nox.session(name='docs', python=False)
def run_sphinx(session: nox.Session) -> None:
    """
    Run spellcheck and then use sphinx to build docs

    Use the optional 'clean' argument to remove everything under the 'build'
    and 'source/api' folders prior to re-building the docs. This is important
    in cases where the api module names have been changed or when some navbars
    are not showing new pages. In general, try without 'clean' first.

    """
    if 'clean' in session.posargs:
        os.chdir('docs')
        session.run('make', 'clean')

        if os.path.exists('source/api'):
            shutil.rmtree('source/api')

        if os.path.exists('jupyter_execute'):
            shutil.rmtree('jupyter_execute')

        os.chdir('..')

    run_spellcheck(session)

    session.run('sphinx-build', 'docs/source', 'docs/build')


@nox.session(name='pre-commit', python=False)
def run_pre_commit(session: nox.Session) -> None:
    """
    Run all linters/tests and make new badges

    Order of sessions: ruff, spellcheck, pytest, genbadge. Using 'format' for
    linter, 'write' for spellcheck, and/or 'parallel' for pytest is permitted.

    """
    run_ruff(session)
    run_spellcheck(session)

    run_pytest(session)
    run_genbadge(session)


@nox.session(name='rebuild', python=False)
def run_build_ext(session: nox.Session) -> None:
    """
    Rebuilds the Cython extensions in place

    Use this to rebuild your editable package when the Cython files change.
    This removes the need to fully reinstall. Note, this only works if your
    original install was done in editable mode. Editable mode already takes
    care of keeping modified python files synced, so only rebuild when any
    extension files (.pyx, .pxd) are updated.

    """
    session.run('pip', 'install', '--upgrade', '--quiet', 'cython')
    session.run('python', 'setup.py', 'build_ext', '--inplace')

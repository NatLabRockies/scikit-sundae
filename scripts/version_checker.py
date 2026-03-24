import requests
import argparse

from packaging.version import Version


def get_latest_version(package: str, prefix: str = None) -> str:
    """
    Fetch the latest version with matching prefix from PyPI.

    Parameters
    ----------
    package : str
        The name of the package to query.
    prefix : str
        A filtering prefix used to get a subset of releases, e.g., '1.1' will
        return the latest patch to version 1.1 even if 1.2 exists.

    Returns
    -------
    latest_version : str
        The latest version available on PyPI. '0.0.0' is returned if there
        are no versions.

    Raises
    ------
    ValueError
        Failed to fetch PyPI data for requested package.

    """
    url = f"https://pypi.org/pypi/{package}/json"

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch PyPI data for '{package}'.")

    data = response.json()
    versions = list(data.get('releases', {}).keys())
    if not versions:
        print(f"{package=} not found on PyPI.")
        return '0.0.0'

    if prefix:
        versions = [v for v in versions if v.startswith(prefix)]
        assert len(versions) != 0, f"{prefix=} has no existing matches."

    sorted_versions = sorted(versions, key=Version, reverse=True)
    latest_version = sorted_versions[0]

    print(f"Latest PyPI version for {package}: {latest_version}.")
    return latest_version


def check_against_pypi(pypi: str, local: str) -> None:
    """
    Verify the local version is newer than PyPI.

    Parameters
    ----------
    pypi : str
        Latest version on PyPI.
    local : str
        Local package version.

    Returns
    -------
    None.

    Raises
    ------
    ValueError
        Local package is older than PyPI.

    """
    pypi = Version(pypi)
    local = Version(local)

    if local < pypi:
        raise ValueError(f"Local package {local} is older than PyPI {pypi}.")

    print(f"Local package {local} is newer than PyPI {pypi}.")


def check_against_tag(tag: str, local: str) -> None:
    """
    Check that the tag matches the package version.

    Parameters
    ----------
    tag : str
        Semmantically versioned tag.
    local : str
        Local package version.

    Returns
    -------
    None.

    Raises
    ------
    ValueError
        Version mismatch: tag differs from local.

    """
    tag = Version(tag)
    local = Version(local)

    if tag != local:
        raise ValueError(f"Version mismatch: {tag=} vs. {local=}")

    print(f"Local and tag versions match: {tag} == {local}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()

    check_against_tag(args.tag, args.local)

    patch_check = Version(args.local)
    if patch_check.micro > 0:
        prefix = str(patch_check.major) + '.' + str(patch_check.minor) + '.'
    else:
        prefix = None

    pypi = get_latest_version('scikit-sundae', prefix)

    check_against_pypi(pypi, args.local)

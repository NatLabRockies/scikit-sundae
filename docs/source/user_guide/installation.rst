Installation
============
This page will guide you through the installation process for scikit-SUNDAE. Whether you are looking to install the package via `pip` from PyPI, `conda` from the conda-forge channel, or from the source distribution, this page has you covered. Additionally, troubleshooting tips and instructions for installing the development version of scikit-SUNDAE are provided.

Installing via PyPI
-------------------
Installing with `pip` will pull a distribution file from the Python Package Index (PyPI). We provide both binary and source distributions on `PyPI`_. Default settings will use binary distributions, so long as your hardware and OS combination are supported. See below for a list of unsupported platforms.

To install the latest release, simply run the following::

    pip install scikit-sundae

.. _PyPI: https://pypi.org/project/scikit-sundae

Python Version Support
^^^^^^^^^^^^^^^^^^^^^^
Please note that scikit-SUNDAE releases only support whichever Python versions are actively maintained at the time of the release. If you are using a version of Python that has reached the end of its life, as listed on the `official Python release page`_, you may need to install an older version of scikit-SUNDAE or upgrade your Python version. To install a specific, older version that supports your current Python installation use::

    pip install scikit-sundae==x.x

where `x.x` is replaced with a specific major/minor version number. We recommend, however, upgrading your Python version instead of using an older version of scikit-SUNDAE.

.. _official Python release page: https://devguide.python.org/versions/

Platform Support
^^^^^^^^^^^^^^^^
scikit-SUNDAE currently supports the following architectures:

* 64-bit x86 architectures on all major platforms: MacOS, Linux, and Windows
* ARM support is available for MacOS
* We do not support 32-bit systems, aarch64, or ppc64le

If you are using an unsupported platform, you may still be able to install using the source distribution. See :ref:`Installing from Source` below for more information.

Installing via conda-forge
--------------------------
To install scikit-SUNDAE via `conda`, you must specify the conda-forge channel. You can install the package with the following::

    conda install -c conda-forge scikit-sundae

Python Version Support
^^^^^^^^^^^^^^^^^^^^^^
Please note that scikit-SUNDAE releases only support whichever Python versions are actively maintained at the time of the release. If you are using a version of Python that has reached the end of its life, as listed on the `official Python release page`_, you may need to install an older version of scikit-SUNDAE or upgrade your Python version. To install a specific, older version that supports your current Python installation use::

    conda install -c conda-forge scikit-sundae=x.x

where `x.x` is replaced with a specific major/minor version number. We recommend, however, upgrading your Python version instead of using an older version of scikit-SUNDAE.

Platform Support
^^^^^^^^^^^^^^^^
As with the `pip` installation, the `conda` installation also has limits on supported platforms:

* 64-bit x86 architectures on all major platforms: MacOS, Linux, and Windows
* ARM support is available for MacOS
* We do not support 32-bit systems, aarch64, or ppc64le

If you are using an unsupported platform, you may still be able to install using the source distribution on PyPI. See :ref:`Installing from Source` below for more information.

Install Issues
--------------
In some cases, you may encounter issues during installation. This section covers common problems and steps to resolve them.

.. _Cannot Locate SUNDIALS:

Cannot Locate SUNDIALS
^^^^^^^^^^^^^^^^^^^^^^
If you are not installing from the binary releases, then you may run into an error message like::

    FileNotFoundError("Can't find SUNDIALS installation in default search paths.")

This means that your SUNDIALS installation is not in an expected location, i.e., `$CONDA_PREFIX` or `%CONDA_PREFIX%\Library`. In this case, you will need to search through your drive(s) to determine where SUNDIALS is installed. After locating the parent directory for your SUNDIALS installation, set the `SUNDIALS_PREFIX` environment variable. On Linux-like systems, use::

    export SUNDIALS_PREFIX="..."

Alternative, if you are on a Windows machine, use::

    set SUNDIALS_PREFIX="..."

The parent directory should have both the `include` and `lib` folders from the SUNDIALS installations. For example if both `C:\SUNDIALS\include` and `C:\SUNDIALS\lib` are on your computer, then your parent directory is `C:\SUNDIALS`. After setting the environment variable, try your installation again.

Missing Binaries
^^^^^^^^^^^^^^^^
If pre-built binary distributions are not available for your platform, we encourage you to `submit an issue on GitHub`_ so that we can look into adding it in the future. There are already plans to add ARM support for Windows and Linux, but there is no timeline on when this will be completed. Be aware that adding new platform-specific binaries to our release workflow is not trivial. If you do submit an issue, do not expect a quick fix.

.. _submit an issue on GitHub: https://github.com/NatLabRockies/scikit-sundae/issues

Additionally, you can follow the instructions below to build the package from the source distribution. If you use the conda package manager then this is relatively straightforward. Even if it sounds intimidating, we highly recommend you give it a try to get up and going now rather than waiting for us to "officially" support your platform in a future release. 

.. _Installing from Source:

Installing from Source
----------------------
Installing scikit-SUNDAE from source is recommended only in the following cases:

1. You need to compile against a custom SUNDIALS configuration
2. No binary distribution is available for your specific platform

.. note:: 

    Please be aware that we do not provide installation support for users compiling against custom SUNDIALS builds. Our package is built and tested against releases available on conda-forge, and we cannot guarantee compatibility with custom builds. We simply include this in the installation instructions for those interested. However, you are on your own if the package does not compile against your custom configuration. If you experience issues, we recommend compiling against SUNDIALS releases from conda-forge instead, which we provide support for. 

Instructions
^^^^^^^^^^^^
In all cases where you are trying to install from the source distribution, you should consult the table below. This specifies which version(s) of SUNDIALS are compatible with each version of scikit-SUNDAE. Since the source distribution does not come with a pre-compiled SUNDIALS, you are responsible to make sure that the version you install is compatible, whether you are installing from conda-forge releases, or not. 

====================== ==============================
scikit-SUNDAE Version  Supported SUNDIALS Version(s)
====================== ==============================
1.2.x [dev]             >=7.6, <7.8
1.1.x                   >=7.3, <7.6
1.0.x                   >=7.0, <7.3
====================== ==============================

1. Make sure you have a C compiler installed. If you are using the Windows operating system, we recommend `Visual Studio`_, but if you have another preferred compiler that should work too. For MacOS and other Linux-based operating systems, we recommend `Clang`_, which can be installed through the terminal using::

    xcode-select --install      (for MacOS)
    sudo apt install clang      (for Linux)

2. Skip to step (3) if you are trying to compile against a custom SUNDIALS build. Otherwise, install a supported SUNDIALS release from conda-forge. Reference the table above to find a version that is compatible with the version of scikit-SUNDAE you are trying to install, and fill in the `x.x` below with an appropriate version::

    conda install -c conda-forge sundials=x.x 

3. Skip this step if you installed SUNDIALS from conda-forge during step (2). Otherwise, make sure your SUNDIALS version is compatible with the version of scikit-SUNDAE you're trying to install using the table above as a reference. Afterward, find the parent directory for your SUNDIALS files and follow the directions in the :ref:`Cannot Locate SUNDIALS` section to set the `SUNDIALS_PREFIX` environment variable.

4. Force the package to install the source distribution from PyPI, with the verbose option:: 

    pip install scikit-sundae --no-binary scikit-sundae -v

.. _Visual Studio: https://visualstudio.microsoft.com/
.. _Clang: https://clang.llvm.org/

Developer Versions
------------------
The development version is ONLY hosted on GitHub. To install it, see the :doc:`../development/index` section. You should only do this if you:

* Want to try experimental features
* Need access to unreleased fixes
* Would like to contribute to the package

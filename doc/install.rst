..
    : IODATA is an input and output module for quantum chemistry.
    :
    : Copyright (C) 2011-2019 The IODATA Development Team
    :
    : This file is part of IODATA.
    :
    : IODATA is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : IODATA is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --


Installation
============

Python 3 (>=3.6) must be installed before you can install IOData. In addition,
IOData has the following dependencies:

- numpy >= 1.0: https://numpy.org/
- scipy: https://scipy.org/
- attrs >= 20.1.0: https://www.attrs.org/en/stable/index.html
- importlib_resources [only for Python 3.6]: https://gitlab.com/python-devs/importlib_resources

You only need to install these dependencies manually when installing IOData from
source. We recommend installing IOData as a Conda or PyPI package instead, which
will automatically install the dependencies (other than Python itself). Package
details can be found here:

- https://anaconda.org/theochem/iodata (Conda package)
- https://pypi.org/project/qc-iodata (PyPI package)

To install IOData using the conda package management system, install
`miniconda <https://conda.io/miniconda.html>`__ or
`anaconda <https://www.anaconda.com/download>`__ first, and then:

.. code-block:: bash

    # Activate your main conda environment if it is not loaded in your .bashrc.
    # E.g. run the following if you have miniconda installed in e.g. ~/miniconda3
    source ~/miniconda3/bin/activate

    # Create a horton3 conda environment. (optional, recommended)
    conda create -n horton3
    source activate horton3

    # Install the stable release.
    conda install -c theochem iodata

    # For developers:
    # (Only do this if you understand the implications.)
    # Install the testing release. (beta)
    conda install -c theochem/label/test iodata
    # Install the development release. (alpha)
    conda install -c theochem/label/dev iodata


There are two common methods to install IOData with pip:

1. You can work in a `virtual environment`_:

   .. code-block:: bash

       # Create a virtual environment in ~/horton3
       # Feel free to change the path.
       python3 -m venv ~/horton3

       # Activate the virtual environemnt.
       source ~/horton3/bin/activate

       # Install the stable release in the venv horton3.
       pip3 install qc-iodata
       # alternative: python3 -m pip install qc-iodata

       # For developers, install a pre-release (alpha or beta).
       # (Only do this if you understand the implications.)
       pip3 install --pre qc-iodata
       # alternative: python3 -m pip install --pre qc-iodata

2. You can install into your ``${HOME}`` directory, without creating a virtual
   environment.

   .. code-block:: bash

       # Install the stable release in your home directory.
       pip3 install qc-iodata --user
       # alternative: python3 -m pip install qc-iodata --user

       # For developers, install a pre-release (alpha or beta).
       # (Only do this if you understand the implications.)
       pip3 install --pre qc-iodata --user
       # alternative: python3 -m pip install --pre qc-iodata --user

   This is by far the simplest method, ideal to get started, but you have only
   one home directory. If the installation breaks due to some experimentation,
   it is harder to make a clean start in comparison to the other options.

In case the ``pip3`` executable is not found, pip may be installed in a
directory which is not included in your ``${PATH}`` variable. This seems to be a
common issue on macOS. A simple workaround is to replace ``pip3`` by
``python3 -m pip``.

In case Python and your operating system are up to date, you may also use
``pip`` instead of ``pip3`` or ``python`` instead of ``python3``. The ``3`` is
only used to avoid potential confusion with Python 2. Note that the ``3`` is
only present in names of executables, not names of Python modules.


Testing
-------

The tests are automatically run when we build packages with conda, but you may
try them again on your own machine after installation.

With Conda:

.. code-block:: bash

    # Install pytest in your conda env.
    conda install pytest pytest-xdist
    # Then run the tests.
    pytest --pyargs iodata -n auto


With Pip:

.. code-block:: bash

    # Install pytest in your conda env ...
    pip install pytest pytest-xdist
    # .. and refresh the virtual environment.
    # This is a venv quirk. Without it, pytest may not find IOData.
    deactivate && source ~/horton3/activate

    # Alternatively, install pytest in your home directory.
    pip install pytest pytest-xdist --user

    # Finally, run the tests.
    pytest --pyargs iodata -n auto


.. _virtual environment: https://docs.python.org/3/tutorial/venv.html

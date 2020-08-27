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

Python 3 (>=3.6) must be installed. Other dependencies will be pulled in with
the instructions below.

To install IOData using conda package management system, install
`miniconda <https://conda.io/miniconda.html>`__ or
`anaconda <https://www.anaconda.com/download>`__ first, and then:

.. code-block:: bash

    # Activate your main conda environment if needed.

    # Create a horton3 conda environment. (optional, recommended)
    conda create -n horton3
    source activate horton3

    # Install the stable release.
    conda install -c theochem iodata

    # For developers:
    # Install the testing release. (beta)
    conda install -c theochem/label/test iodata
    # Install the development release. (alpha)
    conda install -c theochem/label/dev iodata

To install IOData with pip, you may want to create a `virtual environment`_,
and then:

.. code-block:: bash

    # Install the stable release.
    pip install qc-iodata

    # For developers, install a pre-release (alpha or beta)
    pip install --pre qc-iodata


Testing
-------

The tests are automatically run when building with conda, but you may try
them again on your own machine after installation:

.. code-block:: bash

    # Install pytest in conda ...
    conda install pytest
    # ... or with pip.
    pip install pytest
    # Then run the tests.
    pytest --pyargs iodata


.. _virtual environment: https://docs.python.org/3/tutorial/venv.html

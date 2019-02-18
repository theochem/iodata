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

Install
=======

Dependencies
------------

IOData (like all HORTON3 packages) is built using conda:

It is provided in the conda **theochem** channel.


Installation (from Conda)
-------------------------

To install IOData:

.. code-block:: bash

    $ conda -c theochem install iodata

.. _install_from_source:

Installation (from source)
--------------------------

If you wish to build from source, you will need the **conda-build** package
to build it.

You must set the PROJECT_VERSION and MYCONDAPY environmental variables to
emulate the travis build environment.

From project root, issue some variation of:

.. code-block:: bash

    $ PROJECT_VERSION=0.0.0 MYCONDAPY=3.7 conda-build -c theochem tools/conda.recipe

Installation (by-hand)
----------------------

Advanced developers may build by hand using the dependencies listed below,
but the procedure is entirely unsupported.

The following dependencies will be necessary for IOData to build properly,

* Python >= 3.6: http://www.python.org/
* SciPy >= 0.11.0: http://www.scipy.org/
* NumPy >= 1.9.1: http://www.numpy.org/
* pytest >= 4.2.0: https://docs.pytest.org/
* Cython
* gcc/clang


Testing
-------

The tests are automatically run when building with conda, but you may try
them again on your own machine:

.. code-block:: bash

    $ pytest -v iodata

Building Docs
-------------

To build the documentation locally (a necessity for any contributions back to master), install
the following additional requirements:

* Sphinx
* sphinx-rtd-theme
* sphinxcontrib-napoleon
* sphinx-autodoc-typehints

FYI, most are not available within Conda. Pip will still happily install them into
an active conda environment though.

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

IOData
======
|Travis|
|Conda|
|Pypi|
|Codecov|
|Version|
|CondaVersion|
|License|


About
-----
IOData is a HORTON 3 module for input/output of quantum chemistry file formats. Documentation is
here: https://iodata.readthedocs.io/en/latest/index.html


Dependencies
------------

The following dependencies will be necessary for IOData to build properly,

* Python >= 3.6: http://www.python.org/
* SciPy >= 0.11.0: http://www.scipy.org/
* NumPy >= 1.9.1: http://www.numpy.org/
* pytest >= 4.2.0: https://docs.pytest.org/


Installation
------------

To install IOData using conda package management system, install
`miniconda <https://conda.io/miniconda.html>`__ or
`anaconda <https://www.anaconda.com/download>`__ first, and then:

.. code-block:: bash

    $ conda create -n iodata_env
    $ source activate iodata_env
    (iodata_env) $ conda install -c theochem iodata

See https://iodata.readthedocs.io/en/latest/install.html for full details.

.. |Travis| image:: https://travis-ci.org/theochem/iodata.svg?branch=master
    :target: https://travis-ci.org/theochem/iodata
.. |Version| image:: https://img.shields.io/pypi/pyversions/iodata.svg
.. |License| image:: https://img.shields.io/github/license/theochem/iodata
.. |Pypi| image:: https://img.shields.io/pypi/v/iodata.svg
    :target: https://pypi.python.org/pypi/iodata/0.1.3
.. |Codecov| image:: https://img.shields.io/codecov/c/github/theochem/iodata/master.svg
    :target: https://codecov.io/gh/theochem/iodata
.. |Conda| image:: https://img.shields.io/conda/v/theochem/iodata.svg
    :target: https://anaconda.org/theochem/iodata
.. |CondaVersion| image:: https://img.shields.io/conda/pn/theochem/iodata.svg
    :target: https://anaconda.org/theochem/iodata

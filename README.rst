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
|pytest|
|release|
|CodeFactor|
|PyPI|
|Version|
|License|


About
-----

IOData is a HORTON 3 module for input/output of quantum chemistry file formats.
Documentation is here: https://iodata.readthedocs.io/en/latest/index.html

Citation
--------

Please use the following citation in any publication using IOData library:

    **"IOData: A python library for reading, writing, and converting computational chemistry file
    formats and generating input files."**, T. Verstraelen, W. Adams, L. Pujal, A. Tehrani, B. D.
    Kelly, L. Macaya, F. Meng, M. Richer, R. Hernandez‐Esparza, X. D. Yang, M. Chan, T. D. Kim, M.
    Cools‐Ceuppens, V. Chuiko, E. Vohringer‐Martinez,P. W. Ayers, F. Heidar‐Zadeh,
    `J Comput Chem. 2021; 42: 458– 464 <https://doi.org/10.1002/jcc.26468>`__.

Installation
------------

..
    : To install IOData using the conda package management system, install
    : `miniconda <https://conda.io/miniconda.html>`__ or
    : `anaconda <https://www.anaconda.com/download>`__ first, and then:
    :
    : .. code-block:: bash
    :
    :     # Create a horton3 conda environment. (optional, recommended)
    :     conda create -n horton3
    :     source activate horton3
    :
    :     # Install the stable release.
    :     conda install -c theochem iodata
    :
    : To install IOData with pip, you may want to create a `virtual environment`_,
    : and then:
    :
    : .. code-block:: bash
    :
    :     # Install the stable release.
    :     pip install qc-iodata

In anticipation of the 1.0 release of IOData, install the latest git revision
as follows:

.. code-block:: bash

    python -m pip install git+https://github.com/theochem/iodata.git

Add the ``--user`` argument if you are not working in a virtual or conda
environment. Note that there may be API changes between subsequent revisions.

See https://iodata.readthedocs.io/en/latest/install.html for full details.

.. |pytest| image:: https://github.com/theochem/iodata/actions/workflows/pytest.yml/badge.svg
    :target: https://github.com/theochem/iodata/actions/workflows/pytest.yml
.. |release| image:: https://github.com/theochem/iodata/actions/workflows/release.yml/badge.svg
    :target: https://github.com/theochem/iodata/actions/workflows/release.yml
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/tovrstra/stepup-core/badge
    :target: https://www.codefactor.io/repository/github/tovrstra/stepup-core
.. |Version| image:: https://img.shields.io/pypi/pyversions/qc-iodata.svg
.. |License| image:: https://img.shields.io/github/license/theochem/iodata
.. |PyPI| image:: https://img.shields.io/pypi/v/qc-iodata.svg
    :target: https://pypi.python.org/pypi/qc-iodata/
.. _virtual environment: https://docs.python.org/3/tutorial/venv.html

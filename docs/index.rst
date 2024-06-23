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

.. IOData documentation master file, created by
   sphinx-quickstart on Thu Oct 11 04:00:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IOData's documentation!
==================================

IOData is a free and open-source Python library for parsing, storing, and
converting various file formats commonly used by quantum chemistry,
molecular dynamics, and plane-wave density-functional-theory software programs.
It also supports a flexible framework for generating input files for various
software packages.

Please use the following citation in any publication using IOData library:

    **"IOData: A python library for reading, writing, and converting computational chemistry file
    formats and generating input files."**, T. Verstraelen, W. Adams, L. Pujal, A. Tehrani, B. D.
    Kelly, L. Macaya, F. Meng, M. Richer, R. Hernandez‐Esparza, X. D. Yang, M. Chan, T. D. Kim, M.
    Cools‐Ceuppens, V. Chuiko, E. Vohringer‐Martinez,P. W. Ayers, F. Heidar‐Zadeh,
    `J Comput Chem. 2021; 42: 458–464 <https://doi.org/10.1002/jcc.26468>`_.

Copy-pasteable citation records in various formats are provided in :ref:`how_to_cite`.

For the list of file formats that can be loaded or dumped by IOData, see
:ref:`file_formats`. The two tables below summarize the file formats and
features supported by IOData.

========= ==========
Code      Definition
========= ==========
**L**     loading is supported
**D**     dumping is supported
*(d)*     attribute may be derived from other attributes
R         attribute is always read
r         attribute is read if present
W         attribute is always written
w         attribute is is written if present
========= ==========

.. include:: formats_tab.inc


User Documentation
^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   install
   how_to_cite
   getting_started/index
   formats
   inputs
   basis
   changelog
   acknowledgments

Developer Documentation
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   contributing
   code_of_conduct

API Reference
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   pyapi/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

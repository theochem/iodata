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

IOData is the HORTON3 module for reading and writing different quantum chemistry formats.

Currently we support the following formats to varying degrees: **XYZ, POSCAR,
Cube, CHGCAR, LOCPOT, Fchk, Molden, MKL, WFN, FCIDUMP, CP2K ATOM output and
Gaussian log**. See :ref:`file_formats` for details. IOData primarily focusses
on correctly reading in wavefunctions from these file formats, where needed also
correcting for common errors in the Molden and Molekel formats introduced by
various programs (ORCA, TurboMole and pre-1.0 versions of PSI4).

User Documentation
^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   install
   getting_started
   formats
   changelog

Developer Documentation
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   contributing

API Reference
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   pyapi/modules


Acknowledgments
===============

This software was developed using funding from a variety of international
sources including, but not limited to: Canarie, the Canada Research Chairs,
Compute Canada, the European Union's Horizon 2020 Marie Sklodowska-Curie grant
(No 800130), the Foundation of Scientific Research--Flanders (FWO), McMaster
University, the National Fund for Scientific and Technological Development of
Chile (FONDECYT), the Natural Sciences and Engineering Research Council of
Canada (NSERC), the Research Board of Ghent University (BOF), and Sharcnet.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

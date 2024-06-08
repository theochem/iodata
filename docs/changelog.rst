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

Changelog
#########

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

`Unreleased`_
=============

Changed
-------

Originally, IOData was a subpackage of HORTON2. It is currently factored out,
modernized and ported to Python 3. In this process, the API was seriously
refactored, essentially designed from scratch. Compared to HORTON2, IOData 1.0.0
contains the following API-breaking changes:

* The user-facing API is now a set of five functions:
  :py:func:`iodata.api.load_one`, :py:func:`iodata.api.dump_one`,
  :py:func:`iodata.api.load_many`, :py:func:`iodata.api.dump_many` and
  :py:func:`iodata.api.write_input`.
* The :py:class:`iodata.iodata.IOData` object is implemented with the
  `attrs <https://www.attrs.org>`_ module, which facilites type hinting and
  checking.
* The ``load_many`` and ``dump_many`` functions can handle trajectories and
  database formats. (At the time of writing, only XYZ and FCHK are supported.)
* The ``write_input`` function can be used to prepare inputs for quantum
  chemistry software. This function supports user-provided templates.
* IOData does not impose a specific ordering of the atomic orbital basis
  functions (within one shell). Practically all possible conventions are
  supported and one can easily convert from one to another.
* All attributes of IOData are either built-in Python types, Numpy arrays or
  NamedTuples defined in IOData. It no longer relies on other parts of HORTON2
  to define these data types. (This is most relevant for the orbital basis,
  the molecular orbitals and the cube data.)
* Nearly all attributes of the IOData class have been renamed to more systematic
  terminology.
* All file format modules have an identical API (and therefore do not fit into
  a single namespace).
* Ghost atoms are now loaded as atoms with a zero effective core charge
  (``atcorenums``).
* Spin multiplicity is no longer used. Instead, the spin polarization is stored
  `= abs(nalpha - nbeta)`.
* The internal HDF5 file format support has been removed.
* Many smaller changes have been made, which would be too tedious to be listed
  here.

In addition, several new attributes were added to the ``IOData`` class, and
several of them can also be read from file formats we already supported
previously. This work will be expanded upon in future releases.


.. _Unreleased: https://github.com/theochem/iodata

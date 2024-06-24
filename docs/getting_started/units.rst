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

.. _units:

Unit conversion
===============

IOData always represents all quantities in atomic units and unit conversion
constants are defined in ``iodata.utils``. Conversion *to* atomic units is done
by *multiplication* with a unit constant. This convention can be easily
remembered with the following examples:

- When you say "this bond length is 1.5 Å", the IOData equivalent is
  ``bond_length = 1.5 * angstrom``.

- The conversion from atomic units is similar to axes labels in old papers.
  For example. a bond length in angstrom is printed as "Bond length / Å".
  Expressing this with IOData's conventions gives
  ``print("Bond length in Angstrom:", bond_length / angstrom)``

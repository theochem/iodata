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

Getting Started
===============

IOData can be used to read and write various quantum chemistry file formats.

The ``iodata-convert`` script can be used for simple conversions.
More complex use cases can be implemented in Python,
allowing you to access all loaded data as Python objects
that can be modified or updated before writing to a new file.


.. toctree::
   :maxdepth: 2

   script
   loading
   dumping
   inputs
   representation
   units

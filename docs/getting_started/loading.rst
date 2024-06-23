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

Loading Files
=============

To read a file, use something like this:

.. literalinclude:: ../example_scripts/load_water.py
    :language: python
    :linenos:
    :lines: 3-

**Note that IOData will automatically convert units from the file format's
official specification to atomic units (which is the format used throughout
HORTON3).**

The file format is inferred from the extension, but one can override the
detection mechanism by manually specifying the format:

.. literalinclude:: ../example_scripts/load_water_foo.py
    :language: python
    :linenos:
    :lines: 3-

IOData also has basic support for loading databases of molecules. For example,
the following will iterate over all frames in an XYZ file:

.. literalinclude:: ../example_scripts/load_trajectory.py
    :language: python
    :linenos:
    :lines: 3-

More details can be found in the API documentation of
:py:func:`iodata.api.load_one` and :py:func:`iodata.api.load_many`.

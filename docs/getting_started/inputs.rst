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

Writing Input Files
===================

IOData can be used to write input files for quantum-chemistry software. By
default minimal settings are used, which can be changed if needed. For example,
the following will prepare a Gaussian input for a HF/STO-3G calculation from
a PDB file:

.. literalinclude:: ../example_scripts/write_gaussian_com.py
    :language: python
    :linenos:
    :lines: 3-

The level of theory and other settings can be modified by setting corresponding
attributes in the IOData object:

.. literalinclude:: ../example_scripts/write_gaussian_com_lot.py
    :language: python
    :linenos:
    :lines: 3-

The run types can be any of the following: ``energy``, ``energy_force``,
``opt``, ``scan`` or ``freq``. These are translated into program-specific
keywords when the file is written.

It is possible to define a custom input file template to allow for specialized
commands. This is done by passing a template string using the optional ``template`` keyword,
placing each IOData attribute (or additional keyword, as shown below) in curly brackets:

.. literalinclude:: ../example_scripts/write_gaussian_com_template.py
    :language: python
    :linenos:
    :lines: 3-

The input file template may also include keywords that are not part of the IOData
object:

.. literalinclude:: ../example_scripts/write_gaussian_com_custom.py
    :language: python
    :linenos:
    :lines: 3-

In some cases, it may be preferable to load the template from file, instead of
defining it in the script:

.. literalinclude:: ../example_scripts/write_gaussian_com_file.py
    :language: python
    :linenos:
    :lines: 3-

More details can be found in the API documentation of
:py:func:`iodata.api.write_input`.

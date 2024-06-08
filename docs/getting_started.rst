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

IOData can be used to read and write different quantum chemistry file formats.


Script usage
------------

The simplest way to use IOData, without writing any code, is to use the ``iodata-convert`` script.

.. code-block:: bash

    iodata-convert input.fchk output.molden

See the :code:`--help` option for more details on usage.


Code usage
----------

More complex use cases can be implemented in Python, using IOData as a library.
IOData stores an object containing the data read from the file.


Reading
^^^^^^^

To read a file, use something like this:

.. literalinclude:: example_scripts/load_water.py
    :language: python
    :linenos:
    :lines: 3-

**Note that IOData will automatically convert units from the file format's
official specification to atomic units (which is the format used throughout
HORTON3).**

The file format is inferred from the extension, but one can override the
detection mechanism by manually specifying the format:

.. literalinclude:: example_scripts/load_water_foo.py
    :language: python
    :linenos:
    :lines: 3-

IOData also has basic support for loading databases of molecules. For example,
the following will iterate over all frames in an XYZ file:

.. literalinclude:: example_scripts/load_trajectory.py
    :language: python
    :linenos:
    :lines: 3-

More details can be found in the API documentation of
:py:func:`iodata.api.load_one` and :py:func:`iodata.api.load_many`.

Writing
^^^^^^^

IOData can also be used to write different file formats:

.. literalinclude:: example_scripts/convert_fchk_molden.py
    :language: python
    :linenos:
    :lines: 3-

One could also convert (and manipulate) an entire trajectory. The following
example converts a geometry optimization trajectory from a Gaussian FCHK file
to an XYZ file:

.. literalinclude:: example_scripts/convert_fchk_xyz_traj.py
    :language: python
    :linenos:
    :lines: 3-

If you wish to perform some manipulations before writing the trajectory, the
simplest way is to load the entire trajectory in a list of IOData objects and
dump it later:

.. literalinclude:: example_scripts/convert_fchk_xyz_traj_mod1.py
    :language: python
    :linenos:
    :lines: 3-

For very large trajectories, you may want to avoid loading it as a whole in
memory. For this, one should avoid making the ``list`` object in the above
example. The following approach would be more memory efficient.

.. literalinclude:: example_scripts/convert_fchk_xyz_traj_mod2.py
    :language: python
    :linenos:
    :lines: 3-

More details can be found in the API documentation of
:py:func:`iodata.api.dump_one` and :py:func:`iodata.api.dump_many`.

Input files
^^^^^^^^^^^

IOData can be used to write input files for quantum-chemistry software. By
default minimal settings are used, which can be changed if needed. For example,
the following will prepare a Gaussian input for a HF/STO-3G calculation from
a PDB file:

.. literalinclude:: example_scripts/write_gaussian_com.py
    :language: python
    :linenos:
    :lines: 3-

The level of theory and other settings can be modified by setting corresponding
attributes in the IOData object:

.. literalinclude:: example_scripts/write_gaussian_com_lot.py
    :language: python
    :linenos:
    :lines: 3-

The run types can be any of the following: ``energy``, ``energy_force``,
``opt``, ``scan`` or ``freq``. These are translated into program-specific
keywords when the file is written.

It is possible to define a custom input file template to allow for specialized
commands. This is done by passing a template string using the optional ``template`` keyword,
placing each IOData attribute (or additional keyword, as shown below) in curly brackets:

.. literalinclude:: example_scripts/write_gaussian_com_template.py
    :language: python
    :linenos:
    :lines: 3-

The input file template may also include keywords that are not part of the IOData
object:

.. literalinclude:: example_scripts/write_gaussian_com_custom.py
    :language: python
    :linenos:
    :lines: 3-

In some cases, it may be preferable to load the template from file, instead of
defining it in the script:

.. literalinclude:: example_scripts/write_gaussian_com_file.py
    :language: python
    :linenos:
    :lines: 3-

More details can be found in the API documentation of
:py:func:`iodata.api.write_input`.

Data representation
^^^^^^^^^^^^^^^^^^^

IOData can be used to represent data in a consistent format for writing at a future point.

.. literalinclude:: example_scripts/data_representation.py
    :language: python
    :linenos:
    :lines: 3-

All supported attributes can be found in the API documentation of the :py:class:`iodata.iodata.IOData` class.

.. _units:

Unit conversion
^^^^^^^^^^^^^^^

IOData always represents all quantities in atomic units and unit conversion
constants are defined in ``iodata.utils``. Conversion *to* atomic units is done
by *multiplication* with a unit constant. This convention can be easily
remembered with the following examples:

- When you say "this bond length is 1.5 Å", the IOData equivalent is
  ``bond_length = 1.5 * angstrom``.

- The conversion from atomic units is similar to axes labels in old papers.
  For example. a bond length in angstrom is printed as "Bond length / Å".
  Expressing this with IOData's conventions gives
  ``print("Bond length in Angstrom:", bond_length /  angstrom)``

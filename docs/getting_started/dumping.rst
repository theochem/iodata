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

Dumping Files
=============

IOData can also be used to write different file formats:

.. literalinclude:: ../example_scripts/convert_fchk_molden.py
    :language: python
    :linenos:
    :lines: 3-

One could also convert (and manipulate) an entire trajectory. The following
example converts a geometry optimization trajectory from a Gaussian FCHK file
to an XYZ file:

.. literalinclude:: ../example_scripts/convert_fchk_xyz_traj.py
    :language: python
    :linenos:
    :lines: 3-

If you wish to perform some manipulations before writing the trajectory, the
simplest way is to load the entire trajectory in a list of IOData objects and
dump it later:

.. literalinclude:: ../example_scripts/convert_fchk_xyz_traj_mod1.py
    :language: python
    :linenos:
    :lines: 3-

For very large trajectories, you may want to avoid loading it as a whole in
memory. For this, one should avoid making the ``list`` object in the above
example. The following approach would be more memory efficient.

.. literalinclude:: ../example_scripts/convert_fchk_xyz_traj_mod2.py
    :language: python
    :linenos:
    :lines: 3-

More details can be found in the API documentation of
:py:func:`iodata.api.dump_one` and :py:func:`iodata.api.dump_many`.

# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
'''XYZ file format'''


import numpy as np

from horton.units import angstrom
from horton.periodic import periodic


__all__ = ['load_xyz', 'dump_xyz']


def load_xyz(filename):
    '''Load a molecular geometry from a .xyz file.

       **Argument:**

       filename
            The file to load the geometry from

       **Returns:** dictionary with ``title`, ``coordinates`` and ``numbers``.
    '''
    f = file(filename)
    size = int(f.next())
    title = f.next().strip()
    coordinates = np.empty((size, 3), float)
    numbers = np.empty(size, int)
    for i in xrange(size):
        words = f.next().split()
        numbers[i] = periodic[words[0]].number
        coordinates[i,0] = float(words[1])*angstrom
        coordinates[i,1] = float(words[2])*angstrom
        coordinates[i,2] = float(words[3])*angstrom
    f.close()
    return {
        'title': title,
        'coordinates': coordinates,
        'numbers': numbers
    }


def dump_xyz(filename, data):
    '''Write an ``.xyz`` file.

       **Arguments:**

       filename
            The name of the file to be written. This usually the extension
            ".xyz".

       data
            An IOData instance. Must contain ``coordinates`` and ``numbers``.
            May contain ``title``.
    '''
    with open(filename, 'w') as f:
        print >> f, data.natom
        print >> f, getattr(data, 'title', 'Created with HORTON')
        for i in xrange(data.natom):
            n = periodic[data.numbers[i]].symbol
            x, y, z = data.coordinates[i]/angstrom
            print >> f, '%2s %15.10f %15.10f %15.10f' % (n, x, y, z)

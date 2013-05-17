# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--


import numpy as np

from horton.units import angstrom
from horton.periodic import periodic


__all__ = ['load_geom_xyz']


def load_geom_xyz(filename):
    '''Load a molecular geometry from a .xyz file.

       **Argument:**

       filename
            The file to load the geometry from

       **Returns:** two arrays, coordinates and numbers that can be used as the
       two first arguments of the System constructor.
    '''
    f = file(filename)
    size = int(f.next())
    f.next()
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
        'coordinates': coordinates,
        'numbers': numbers
    }


def dump_xyz(filename, system):
    '''Write a system to a .xyz file.

       **Arguments:**

       filename
            The name of the file to be written. This usually the extension
            ".xyz".

       system
            An instance of the System class.
    '''
    with open(filename, 'w') as f:
        print >> f, system.natom
        print >> f, 'File generated with Horton'
        for i in xrange(system.natom):
            n = periodic[system.numbers[i]].symbol
            x, y, z = system.coordinates[i]/angstrom
            print >> f, '%2s %15.10f %15.10f %15.10f' % (n, x, y, z)

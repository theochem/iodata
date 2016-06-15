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
'''VASP POSCAR, CHGCAR and POTCAR file formats'''


import numpy as np
from horton.units import angstrom, electronvolt
from horton.periodic import periodic
from horton.cext import Cell
from horton.grid.cext import UniformGrid


__all__ = ['load_chgcar', 'load_locpot', 'load_poscar', 'dump_poscar']


def _unravel_counter(counter, shape):
    result = []
    for i in xrange(0, len(shape)):
        result.append(counter % shape[i])
        counter /= shape[i]
    return result


def _load_vasp_header(f, nskip):
    '''Load the cell and atoms from a VASP file

       **Arguments:**

       f
            An open file object

       nskip
            The number of lines to skip after the line with elements

       **Returns:** ``title``, ``cell``, ``numbers``, ``coordinates``
    '''
    # reat the title
    title = f.next().strip()
    f.next()

    # read cell parameters in angstrom. each row is one cell vector
    rvecs = []
    for i in xrange(3):
        rvecs.append([float(w) for w in f.next().split()])
    rvecs = np.array(rvecs)*angstrom

    # Convert to cell object
    cell = Cell(rvecs)

    vasp_numbers = [periodic[w].number for w in f.next().split()]
    vasp_counts = [int(w) for w in f.next().split()]
    numbers = []
    for n, c in zip(vasp_numbers, vasp_counts):
        numbers.extend([n]*c)
    numbers = np.array(numbers)

    # skip some lines
    for i in xrange(nskip):
        f.next()
    assert f.next().startswith('Direct')

    # read the fractional coordinates and convert to Cartesian
    coordinates = []
    for line in f:
        if len(line.strip()) == 0:
            break
        coordinates.append([float(w) for w in line.split()[:3]])
    coordinates = np.dot(np.array(coordinates), rvecs)

    return title, cell, numbers, coordinates


def _load_vasp_grid(filename):
    '''Load a grid data file from VASP 5

       **Arguments:**

       filename
            The VASP filename

       **Returns:** a dictionary containing: ``title``, ``coordinates``,
       ``numbers``, ``cell``, ``grid``, ``cube_data``.
    '''
    with open(filename) as f:
        # Load header
        title, cell, numbers, coordinates = _load_vasp_header(f, 0)

        # read the shape of the data
        shape = np.array([int(w) for w in f.next().split()])

        # read data
        cube_data = np.zeros(shape, float)
        counter = 0
        for line in f:
            if counter >= cube_data.size:
                break
            for w in line.split():
                i0, i1, i2 = _unravel_counter(counter, shape)
                # Fill in the data with transposed indexes. In horton, X is
                # the slowest index while Z is the fastest.
                cube_data[i0, i1, i2] = float(w)
                counter += 1
        assert counter == cube_data.size

    return {
        'title': title,
        'coordinates': coordinates,
        'numbers': numbers,
        'cell': cell,
        'grid': UniformGrid(np.zeros(3), cell.rvecs/shape.reshape(-1,1), shape, np.ones(3, int)),
        'cube_data': cube_data,
    }


def load_chgcar(filename):
    '''Reads a vasp 5 chgcar file.

       **Arguments:**

       filename
            The VASP filename

       **Returns:** a dictionary containing: ``title``, ``coordinates``,
       ``numbers``, ``cell``, ``grid``, ``cube_data``.
    '''
    result = _load_vasp_grid(filename)
    # renormalize electron density
    result['cube_data'] /= result['cell'].volume
    return result


def load_locpot(filename):
    '''Reads a vasp 5 locpot file.

       **Arguments:**

       filename
            The VASP filename

       **Returns:** a dictionary containing: ``title``, ``coordinates``,
       ``numbers``, ``cell``, ``grid``, ``cube_data``.
    '''
    result = _load_vasp_grid(filename)
    # convert locpot to atomic units
    result['cube_data'] *= electronvolt
    return result


def load_poscar(filename):
    '''Reads a vasp 5 poscar file.

       **Arguments:**

       filename
            The VASP filename

       **Returns:** a dictionary containing: ``title``, ``coordinates``,
       ``numbers``, ``cell``.
    '''
    with open(filename) as f:
        # Load header
        title, cell, numbers, coordinates = _load_vasp_header(f, 1)
        return {
            'title': title,
            'coordinates': coordinates,
            'numbers': numbers,
            'cell': cell,
        }


def dump_poscar(filename, data):
    '''Write a file in VASP's POSCAR format

       **Arguments:**

       filename
            The name of the file to be written. This is usually POSCAR.

       data
            An IOData instance. Must contain ``coordinates``, ``numbers``,
            ``cell``. May contain ``title``.
    '''
    with open(filename, 'w') as f:
        print >> f, getattr(data, 'title', 'Created with HORTON')
        print >> f, '   1.00000000000000'

        # Write cell vectors, each row is one vector in angstrom:
        rvecs = data.cell.rvecs
        for rvec in rvecs:
            print >> f, '  % 21.16f % 21.16f % 21.16f' % tuple(rvec/angstrom)

        # Construct list of elements to make sure the coordinates get written
        # in this order. Heaviest elements are put furst.
        unumbers = sorted(np.unique(data.numbers))[::-1]
        print >> f, ' '.join('%5s' % periodic[unumber].symbol for unumber in unumbers)
        print >> f, ' '.join('%5i' % (data.numbers == unumber).sum() for unumber in unumbers)
        print >> f, 'Selective dynamics'
        print >> f, 'Direct'

        # Write the coordinates
        for unumber in unumbers:
            indexes = (data.numbers == unumber).nonzero()[0]
            for index in indexes:
                row = data.cell.to_frac(data.coordinates[index])
                print >> f, '  % 21.16f % 21.16f % 21.16f   F   F   F' % tuple(row)

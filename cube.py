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
from horton.cext import Cell
from horton.grid.cext import UniformGrid


__all__ = ['load_cube', 'dump_cube']


def _read_cube_header(f):
    # skip the first two lines
    f.readline()
    f.readline()

    def read_grid_line(line):
        """Read a grid line from the cube file"""
        words = line.split()
        return (
            int(words[0]),
            np.array([float(words[1]), float(words[2]), float(words[3])], float)
            # all coordinates in a cube file are in atomic units
        )

    # number of atoms and origin of the grid
    natom, origin = read_grid_line(f.readline())
    # numer of grid points in A direction and step vector A, and so on
    shape0, axis0 = read_grid_line(f.readline())
    shape1, axis1 = read_grid_line(f.readline())
    shape2, axis2 = read_grid_line(f.readline())
    shape = np.array([shape0, shape1, shape2], int)
    axes = np.array([axis0, axis1, axis2])

    cell = Cell(axes*shape.reshape(-1,1))
    ugrid = UniformGrid(origin, axes, shape, np.ones(3, int))

    def read_coordinate_line(line):
        """Read an atom number and coordinate from the cube file"""
        words = line.split()
        return (
            int(words[0]), float(words[1]),
            np.array([float(words[2]), float(words[3]), float(words[4])], float)
            # all coordinates in a cube file are in atomic units
        )

    numbers = np.zeros(natom, int)
    pseudo_numbers = np.zeros(natom, float)
    coordinates = np.zeros((natom, 3), float)
    for i in xrange(natom):
        numbers[i], pseudo_numbers[i], coordinates[i] = read_coordinate_line(f.readline())

    return coordinates, numbers, cell, ugrid, pseudo_numbers


def _read_cube_data(f, ugrid):
    data = np.zeros(tuple(ugrid.shape), float)
    tmp = data.ravel()
    counter = 0
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        words = line.split()
        for word in words:
            tmp[counter] = float(word)
            counter += 1
    return data


def load_cube(filename):
    with open(filename) as f:
        coordinates, numbers, cell, ugrid, pseudo_numbers = _read_cube_header(f)
        data = _read_cube_data(f, ugrid)
        props = {
            'ugrid': ugrid,
            'cube_data': data,
        }
        return {
            'coordinates': coordinates,
            'numbers': numbers,
            'cell': cell,
            'props': props,
            'pseudo_numbers': pseudo_numbers,
        }


def _write_cube_header(f, coordinates, numbers, ugrid, pseudo_numbers):
    print >> f, 'Cube file created with Horton'
    print >> f, 'OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z'
    natom = len(numbers)
    x, y, z = ugrid.origin
    print >> f, '%5i % 11.6f % 11.6f % 11.6f' % (natom, x, y, z)
    rvecs = ugrid.grid_cell.rvecs
    for i in xrange(3):
        x, y, z = rvecs[i]
        print >> f, '%5i % 11.6f % 11.6f % 11.6f' % (ugrid.shape[i], x, y, z)
    for i in xrange(natom):
        q = pseudo_numbers[i]
        x, y, z = coordinates[i]
        print >> f, '%5i % 11.6f % 11.6f % 11.6f % 11.6f' % (numbers[i], q, x, y, z)


def _write_cube_data(f, cube_data):
    counter = 0
    for value in cube_data.flat:
        f.write(' % 12.5E' % value)
        if counter%6 == 5:
            f.write('\n')
        counter += 1


def dump_cube(filename, system):
    with open(filename, 'w') as f:
        ugrid = system.props.get('ugrid')
        if ugrid is None:
            raise ValueError('A uniform integration grid must be defined in the system properties (ugrid).')
        cube_data = system.props.get('cube_data')
        if cube_data is None:
            raise ValueError('A cube data array must be defined in the system properties (cube_data).')
        _write_cube_header(f, system.coordinates, system.numbers, ugrid, system.pseudo_numbers)
        _write_cube_data(f, cube_data)

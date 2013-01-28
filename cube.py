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
from horton.grid.uniform import UniformIntGrid


__all__ = ['load_cube']


def read_cube_header(f):
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
    nrep0, axis0 = read_grid_line(f.readline())
    nrep1, axis1 = read_grid_line(f.readline())
    nrep2, axis2 = read_grid_line(f.readline())
    nrep = np.array([nrep0, nrep1, nrep2], int)
    axes = np.array([axis0, axis1, axis2])

    cell = Cell(axes*nrep.reshape(-1,1))
    ui_grid = UniformIntGrid(origin, cell, nrep)

    def read_coordinate_line(line):
        """Read an atom number and coordinate from the cube file"""
        words = line.split()
        return (
            int(words[0]), float(words[1]),
            np.array([float(words[2]), float(words[3]), float(words[4])], float)
            # all coordinates in a cube file are in atomic units
        )

    numbers = np.zeros(natom, int)
    nuclear_charges = np.zeros(natom, float)
    coordinates = np.zeros((natom, 3), float)
    for i in xrange(natom):
        numbers[i], nuclear_charges[i], coordinates[i] = read_coordinate_line(f.readline())

    return coordinates, numbers, ui_grid, nuclear_charges


def read_cube_data(f, ui_grid):
    data = np.zeros(tuple(ui_grid.nrep), float)
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
        coordinates, numbers, ui_grid, nuclear_charges = read_cube_header(f)
        data = read_cube_data(f, ui_grid)
        props = {
            'ui_grid': ui_grid,
            'nuclear_charges': nuclear_charges,
            'cube_data': data,
        }
        return coordinates, numbers, props

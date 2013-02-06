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
from horton.cext import Cell
from horton.grid.uniform import UniformIntGrid


__all__ = ['load_chgcar']



def load_chgcar(filename):
    '''Reads a vasp 5 chgcar file. Not tested extensively yet!'''
    with open(filename) as f:
        # skip first two lines
        f.next()
        f.next()

        # read cell parameters in angstrom
        rvecs = []
        for i in xrange(3):
            rvecs.append([float(w) for w in f.next().split()])
        rvecs = np.array(rvecs)*angstrom

        # compute volume
        cell = Cell(rvecs)

        vasp_numbers = [periodic[w].number for w in f.next().split()]
        vasp_counts = [int(w) for w in f.next().split()]
        numbers = []
        for n, c in zip(vasp_numbers, vasp_counts):
            numbers.extend([n]*c)
        numbers = np.array(numbers)

        # skip one line
        f.next()

        # read the coordinates
        coordinates = []
        for line in f:
            if len(line.strip()) == 0:
                break
            coordinates.append([float(w) for w in line.split()])
        coordinates = np.dot(np.array(coordinates), rvecs.T)

        # read the shape of the data
        shape = np.array([int(w) for w in f.next().split()])

        # read data
        cube_data = np.zeros(shape, float)
        tmp = cube_data.ravel()
        counter = 0
        for line in f:
            if line.startswith('augment'):
                break
            for w in line.split():
                tmp[counter] = float(w)
                counter += 1
        assert counter == tmp.size

        # transpose the cube to make z fastest and x slowest index.
        cube_data = cube_data.transpose(2,1,0)/cell.volume

    props = {
        'ui_grid': UniformIntGrid(np.zeros(3), Cell(rvecs/shape.reshape(-1,1)), shape),
        'cube_data': cube_data}
    return coordinates, numbers, cell, props

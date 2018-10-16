# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
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
"""Gaussian cube file format"""


from __future__ import print_function, annotations

from typing import TextIO, Dict, Tuple, Union

import numpy as np

__all__ = ['load_cube', 'dump_cube']


def _read_cube_header(f: TextIO) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                                          Dict[str, np.ndarray], np.ndarray]:
    # Read the title
    title = f.readline().strip()
    # skip the second line
    f.readline()

    def read_grid_line(line: str) -> Tuple[int, np.ndarray]:
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

    cell = axes * shape.reshape(-1, 1)
    ugrid = {"origin": origin, 'grid_rvecs': axes, 'shape': shape, 'pbc': np.ones(3, int)}

    def read_coordinate_line(line: str) -> Tuple[int, float, np.ndarray]:
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
    for i in range(natom):
        numbers[i], pseudo_numbers[i], coordinates[i] = read_coordinate_line(f.readline())
        # If the pseudo_number field is zero, we assume that no effective core
        # potentials were used.
        if pseudo_numbers[i] == 0.0:
            pseudo_numbers[i] = numbers[i]

    return title, coordinates, numbers, cell, ugrid, pseudo_numbers


def _read_cube_data(f: TextIO, ugrid: Dict[str, np.ndarray]) -> np.ndarray:
    data = np.zeros(tuple(ugrid["shape"]), float)
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


def load_cube(filename: str) -> Dict[str, Union[str, np.ndarray,Dict]]:
    """Load data from a cube file

       Parameters
       ----------
       filename
            The name of the cube file

        Returns
        -------
        Dict
            Contains keys ``title``, ``coordinates``, ``numbers``, ``cell``,
           ``cube_data``, ``grid``, ``pseudo_numbers``.
    """
    with open(filename) as f:
        title, coordinates, numbers, cell, ugrid, pseudo_numbers = _read_cube_header(f)
        data = _read_cube_data(f, ugrid)
        return {
            'title': title,
            'coordinates': coordinates,
            'numbers': numbers,
            'cell': cell,
            'cube_data': data,
            'grid': ugrid,
            'pseudo_numbers': pseudo_numbers,
        }


def _write_cube_header(f: TextIO, title: str, coordinates: np.ndarray, numbers: np.ndarray,
                       ugrid_dict: Dict[str, np.ndarray], pseudo_numbers: np.ndarray):
    print(title, file=f)
    print('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z', file=f)
    natom = len(numbers)
    x, y, z = ugrid_dict["origin"]
    print(f'{natom:5d} {x: 11.6f} {y: 11.6f} {z: 11.6f}', file=f)
    rvecs = ugrid_dict["grid_rvecs"]
    for i in range(3):
        x, y, z = rvecs[i]
        print(f'{ugrid_dict["shape"][i]:5d} {x: 11.6f} {y: 11.6f} {z: 11.6f}', file=f)
    for i in range(natom):
        q = pseudo_numbers[i]
        x, y, z = coordinates[i]
        print(f'{numbers[i]:5d} {q: 11.6f} {x: 11.6f} {y: 11.6f} {z: 11.6f}', file=f)


def _write_cube_data(f: TextIO, cube_data: np.ndarray):
    counter = 0
    for value in cube_data.flat:
        f.write(f' {value: 12.5E}')
        if counter % 6 == 5:
            f.write('\n')
        counter += 1


def dump_cube(filename: str, data: IOData):
    """Write a IOData to a .cube file.

       Parameters
       ----------
       filename
            The name of the file to be written. This usually the extension
            ".cube".

       data
            Must contain ``coordinates``, ``numbers``,
            ``grid``, ``cube_data``. May contain ``title``, ``pseudo_numbers``.
    """
    with open(filename, 'w') as f:
        if not isinstance(data.grid, dict):
            raise ValueError(
                'The system grid must contain a dict to initialize a UniformGrid instance.')
        title = getattr(data, 'title', 'Created with HORTON')
        _write_cube_header(f, title, data.coordinates, data.numbers, data.grid, data.pseudo_numbers)
        _write_cube_data(f, data.cube_data)

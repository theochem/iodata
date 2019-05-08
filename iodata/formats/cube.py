# IODATA is an input and output module for quantum chemistry.
# Copyright (C) 2011-2019 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Module for handling GAUSSIAN CUBE file format."""


from typing import TextIO, Dict, Tuple

import numpy as np

from ..utils import LineIterator


__all__ = []


patterns = ['*.cube']


def _read_cube_header(lit: LineIterator) \
        -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Load header data from a CUBE file object.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        The output tuple contains title, coordinates, atnums, cell, ugrid &
        atcorenums.

    """
    # Read the title
    title = next(lit).strip()
    # skip the second line
    next(lit)

    def read_grid_line(line: str) -> Tuple[int, np.ndarray]:
        """Read a grid line from the cube file."""
        words = line.split()
        return (
            int(words[0]),
            np.array([float(words[1]), float(words[2]), float(words[3])], float)
            # all coordinates in a cube file are in atomic units
        )

    # number of atoms and origin of the grid
    natom, origin = read_grid_line(next(lit))
    # numer of grid points in A direction and step vector A, and so on
    shape0, axis0 = read_grid_line(next(lit))
    shape1, axis1 = read_grid_line(next(lit))
    shape2, axis2 = read_grid_line(next(lit))
    shape = np.array([shape0, shape1, shape2], int)
    axes = np.array([axis0, axis1, axis2])

    cell = axes * shape.reshape(-1, 1)
    ugrid = {"origin": origin, 'grid_rvecs': axes, 'shape': shape, 'pbc': np.ones(3, int)}

    def read_coordinate_line(line: str) -> Tuple[int, float, np.ndarray]:
        """Read an atomic number and coordinate from the cube file."""
        words = line.split()
        return (
            int(words[0]), float(words[1]),
            np.array([float(words[2]), float(words[3]), float(words[4])], float)
            # all coordinates in a cube file are in atomic units
        )

    atnums = np.zeros(natom, int)
    atcorenums = np.zeros(natom, float)
    coordinates = np.zeros((natom, 3), float)
    for i in range(natom):
        atnums[i], atcorenums[i], coordinates[i] = read_coordinate_line(next(lit))
        # If the atcorenum field is zero, we assume that no effective core
        # potentials were used.
        if atcorenums[i] == 0.0:
            atcorenums[i] = atnums[i]

    return title, coordinates, atnums, cell, ugrid, atcorenums


def _read_cube_data(lit: LineIterator, ugrid: Dict[str, np.ndarray]) -> np.ndarray:
    """Load cube data from a CUBE file object.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        The cube data array.

    """
    data = np.zeros(tuple(ugrid["shape"]), float)
    tmp = data.ravel()
    counter = 0
    words = []
    while counter < tmp.size:
        if not words:
            words = next(lit).split()
        tmp[counter] = float(words.pop(0))
        counter += 1
    return data


def load(lit: LineIterator) -> dict:
    """Load data from a CUBE file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        Output dictionary containing ``title``, ``coordinates``, ``atnums``, ``atcorenums``,
        ``cell``, ``cube_data`` & ``grid`` keys and their corresponding values.

    """
    title, coordinates, atnums, cell, ugrid, atcorenums = _read_cube_header(lit)
    data = _read_cube_data(lit, ugrid)
    return {
        'title': title,
        'coordinates': coordinates,
        'atnums': atnums,
        'cell': cell,
        'cube_data': data,
        'grid': ugrid,
        'atcorenums': atcorenums,
    }


def _write_cube_header(f: TextIO, title: str, coordinates: np.ndarray, atnums: np.ndarray,
                       ugrid_dict: Dict[str, np.ndarray], atcorenums: np.ndarray):
    print(title, file=f)
    print('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z', file=f)
    natom = len(atnums)
    x, y, z = ugrid_dict["origin"]
    print(f'{natom:5d} {x: 11.6f} {y: 11.6f} {z: 11.6f}', file=f)
    rvecs = ugrid_dict["grid_rvecs"]
    for i in range(3):
        x, y, z = rvecs[i]
        print(f'{ugrid_dict["shape"][i]:5d} {x: 11.6f} {y: 11.6f} {z: 11.6f}', file=f)
    for i in range(natom):
        q = atcorenums[i]
        x, y, z = coordinates[i]
        print(f'{atnums[i]:5d} {q: 11.6f} {x: 11.6f} {y: 11.6f} {z: 11.6f}', file=f)


def _write_cube_data(f: TextIO, cube_data: np.ndarray):
    counter = 0
    for value in cube_data.flat:
        f.write(f' {value: 12.5E}')
        if counter % 6 == 5:
            f.write('\n')
        counter += 1


def dump(f: TextIO, data: 'IOData'):
    """Write data into a CUBE file format.

    Parameters
    ----------
    f
        A file to write to.
    data
        An IOData instance which must contain ``coordinates``, ``atnums``, ``grid`` &
        ``cube_data`` attributes. It may contain ``title``  & ``atcorenums`` attributes.

    """
    if not isinstance(data.grid, dict):
        raise ValueError(
            'The system grid must contain a dict to initialize a UniformGrid instance.')
    title = getattr(data, 'title', 'Created with HORTON')
    _write_cube_header(f, title, data.coordinates, data.atnums, data.grid, data.atcorenums)
    _write_cube_data(f, data.cube_data)

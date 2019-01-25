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
# pragma pylint: disable=invalid-name
"""Module for handling VASP CHGCAR file format."""


import numpy as np

from typing import TextIO, Tuple, Dict

from .periodic import sym2num
from .utils import angstrom, volume


__all__ = ['load']


patterns = ['CHGCAR*', 'AECCAR*']


def _unravel_counter(counter, shape):
    result = []
    for i in range(0, len(shape)):
        result.append(int(counter % shape[i]))
        counter /= shape[i]
    return result


def _load_vasp_header(f: TextIO) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Load the cell and atoms from a VASP file format.

    Parameters
    ----------
    f
        A VASP file object (in read mode).

    Returns
    -------
    out : tuple
        Output Contains ``title``, ``cell``, ``numbers``, ``coordinates``.

    Notes
    -----
    For file specification see http://cms.mpi.univie.ac.at/vasp/guide/node59.html.

    """
    # read the title
    title = next(f).strip()
    # read the universal scaling factor
    scaling = float(next(f).strip())

    # read cell parameters in angstrom, without the universal scaling factor.
    # each row is one cell vector
    rvecs = []
    for i in range(3):
        rvecs.append([float(w) for w in next(f).split()])
    rvecs = np.array(rvecs) * angstrom * scaling

    # note that in older VASP version the following line might be absent
    vasp_numbers = [sym2num[w] for w in next(f).split()]
    vasp_counts = [int(w) for w in next(f).split()]
    numbers = []
    for n, c in zip(vasp_numbers, vasp_counts):
        numbers.extend([n] * c)
    numbers = np.array(numbers)

    line = next(f)
    # the 7th line can optionally indicate selective dynamics
    if line[0].lower() in ['s']:
        line = next(f)
    # parse direct/cartesian switch
    cartesian = line[0].lower() in ['c', 'k']

    # read the coordinates
    coordinates = []
    for line in f:
        # check if all coordinates are read
        if (len(line.strip()) == 0) or (len(coordinates) == numbers.shape[0]):
            break
        coordinates.append([float(w) for w in line.split()[:3]])
    if cartesian:
        coordinates = np.array(coordinates) * angstrom * scaling
    else:
        coordinates = np.dot(np.array(coordinates), rvecs)

    return title, rvecs, numbers, coordinates


def _load_vasp_grid(filename: str) -> Dict:
    """Load grid data file from the VASP 5 file format.

    Parameters
    ----------
    filename
        The VASP 5 filename.

    Returns
    -------
    out : dict
        Output dictionary containing ``title``, ``coordinates``, ``numbers``, ``rvecs``,
        ``grid`` & ``cube_data`` keys and their corresponding values.

    """
    with open(filename) as f:
        # Load header
        title, rvecs, numbers, coordinates = _load_vasp_header(f)

        # read the shape of the data
        shape = np.array([int(w) for w in next(f).split()])

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

    ugrid = {"origin": np.zeros(3), 'grid_rvecs': rvecs / shape.reshape(-1, 1), 'shape': shape,
             'pbc': np.ones(3, int)}

    return {
        'title': title,
        'coordinates': coordinates,
        'numbers': numbers,
        'rvecs': rvecs,
        'grid': ugrid,
        'cube_data': cube_data,
    }


def load(filename: str) -> Dict:
    """Load data from a VASP 5 CHGCAR file format.

    Parameters
    ----------
    filename
        The VASP 5 CHGCAR filename.

    Returns
    -------
    out : dict
        Output dictionary containing ``title``, ``coordinates``, ``numbers``, ``rvecs``,
        ``grid`` & ``cube_data`` keys and corresponding values.

    """
    result = _load_vasp_grid(filename)
    # renormalize electron density
    result['cube_data'] /= volume(result['rvecs'])
    return result

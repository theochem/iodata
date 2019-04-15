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
"""Module for handling VASP CHGCAR file format."""


from typing import Tuple, Dict

import numpy as np

from ..periodic import sym2num
from ..utils import angstrom, volume, LineIterator


__all__ = []


patterns = ['CHGCAR*', 'AECCAR*']


def _load_vasp_header(lit: LineIterator) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Load the cell and atoms from a VASP file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : tuple
        Output Contains ``title``, ``cell``, ``numbers``, ``coordinates``.

    Notes
    -----
    For file specification see http://cms.mpi.univie.ac.at/vasp/guide/node59.html.

    """
    # read the title
    title = next(lit).strip()
    # read the universal scaling factor
    scaling = float(next(lit).strip())

    # read cell parameters in angstrom, without the universal scaling factor.
    # each row is one cell vector
    rvecs = []
    for _i in range(3):
        rvecs.append([float(w) for w in next(lit).split()])
    rvecs = np.array(rvecs) * angstrom * scaling

    # note that in older VASP version the following line might be absent
    vasp_numbers = [sym2num[w] for w in next(lit).split()]
    vasp_counts = [int(w) for w in next(lit).split()]
    numbers = []
    for n, c in zip(vasp_numbers, vasp_counts):
        numbers.extend([n] * c)
    numbers = np.array(numbers)

    line = next(lit)
    # the 7th line can optionally indicate selective dynamics
    if line[0].lower() in ['s']:
        line = next(lit)
    # parse direct/cartesian switch
    cartesian = line[0].lower() in ['c', 'k']

    # read the coordinates
    coordinates = []
    for _iatom in range(len(numbers)):
        line = next(lit)
        coordinates.append([float(w) for w in line.split()[:3]])
    if cartesian:
        coordinates = np.array(coordinates) * angstrom * scaling
    else:
        coordinates = np.dot(np.array(coordinates), rvecs)

    return title, rvecs, numbers, coordinates


def _load_vasp_grid(lit: LineIterator) -> Dict:
    """Load grid data file from the VASP 5 file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``title``, ``coordinates``, ``numbers``, ``rvecs``,
        ``grid`` & ``cube_data`` keys and their corresponding values.

    """
    # Load header
    title, rvecs, numbers, coordinates = _load_vasp_header(lit)

    # read the shape of the data
    for line in lit:
        shape = np.array([int(w) for w in line.split()])
        if len(shape) == 3:
            break

    # read data
    cube_data = np.zeros(shape, float)
    words = []
    for i2 in range(shape[2]):
        for i1 in range(shape[1]):
            for i0 in range(shape[0]):
                if not words:
                    words = next(lit).split()
                cube_data[i0, i1, i2] = float(words.pop(0))

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


def load(lit: LineIterator) -> Dict:
    """Load data from a VASP 5 CHGCAR file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        Output dictionary containing ``title``, ``coordinates``, ``numbers``, ``rvecs``,
        ``grid`` & ``cube_data`` keys and corresponding values.

    """
    result = _load_vasp_grid(lit)
    # renormalize electron density
    result['cube_data'] /= volume(result['rvecs'])
    return result

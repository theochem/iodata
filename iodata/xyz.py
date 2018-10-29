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
"""Module for handling XYZ file format."""


import numpy as np

from typing import Dict

from .iodata import IOData
from .utils import angstrom
from .periodic import sym2num, num2sym


__all__ = ['load_xyz', 'dump_xyz']


def load_xyz(filename: str) -> Dict:
    """Load a molecular geometry from a .xyz file.

    Parameters
    ----------
    filename
        The file to load the geometry from

    Returns
    -------
    dict
        Contains keys ``title`, ``coordinates`` and ``numbers``.

    """
    with open(filename, 'r') as f:
        size = int(next(f))
        title = next(f).strip()
        coordinates = np.empty((size, 3), float)
        numbers = np.empty(size, int)
        for i in range(size):
            words = next(f).split()
            try:
                numbers[i] = sym2num[words[0].title()]
            except KeyError:
                numbers[i] = int(words[0])
            coordinates[i, 0] = float(words[1]) * angstrom
            coordinates[i, 1] = float(words[2]) * angstrom
            coordinates[i, 2] = float(words[3]) * angstrom
    return {
        'title': title,
        'coordinates': coordinates,
        'numbers': numbers
    }


def dump_xyz(filename: str, data: 'IOData'):
    """Write an ``.xyz`` file.

    Parameters
    ----------
    filename
        The name of the file to be written. This usually the extension
        ".xyz".
    data
        An IOData instance. Must contain ``coordinates`` and ``numbers``.
        May contain ``title``.
    """
    with open(filename, 'w') as f:
        print(data.natom, file=f)
        print(getattr(data, 'title', 'Created with HORTON'), file=f)
        for i in range(data.natom):
            n = num2sym[data.numbers[i]]
            x, y, z = data.coordinates[i] / angstrom
            print(f'{n:2s} {x:15.10f} {y:15.10f} {z:15.10f}', file=f)

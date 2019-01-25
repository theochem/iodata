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
# pragma pylint: disable=wrong-import-order,invalid-name
"""Module for handling XYZ file format."""


import numpy as np

from typing import Dict

from .utils import angstrom
from .periodic import sym2num, num2sym


__all__ = ['load', 'dump']


patterns = ['*.xyz']


def load(filename: str) -> Dict:
    """Load molecular geometry from a XYZ file format.

    Parameters
    ----------
    filename : str
        The XYZ filename.

    Returns
    -------
    out : dict
        Output dictionary containing ``title`, ``coordinates`` & ``numbers`` keys
        and corresponding values.

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


def dump(filename: str, data: 'IOData'):
    """Write molecular geometry into a XYZ file format.

    Parameters
    ----------
    filename : str
        The XYZ filename.
    data : IOData
        An IOData instance which must contain ``coordinates`` & ``numbers`` attributes.
        If ``title`` attribute is not included, 'Created with IODATA module' is used as ``title``.

    """
    with open(filename, 'w') as f:
        print(data.natom, file=f)
        print(getattr(data, 'title', 'Created with IODATA module'), file=f)
        for i in range(data.natom):
            n = num2sym[data.numbers[i]]
            x, y, z = data.coordinates[i] / angstrom
            print(f'{n:2s} {x:15.10f} {y:15.10f} {z:15.10f}', file=f)

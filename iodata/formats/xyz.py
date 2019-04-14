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
"""Module for handling XYZ file format."""


from typing import Dict, TextIO

import numpy as np

from ..utils import angstrom, LineIterator
from ..periodic import sym2num, num2sym


__all__ = []


patterns = ['*.xyz']


def load(lit: LineIterator) -> Dict:
    """Load molecular geometry from a XYZ file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``title`, ``coordinates`` & ``numbers`` keys
        and corresponding values.

    """
    size = int(next(lit))
    title = next(lit).strip()
    coordinates = np.empty((size, 3), float)
    numbers = np.empty(size, int)
    for i in range(size):
        words = next(lit).split()
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


def dump(f: TextIO, data: 'IOData'):
    """Write molecular geometry into a XYZ file format.

    Parameters
    ----------
    f
        A file to write to.
    data : IOData
        An IOData instance which must contain ``coordinates`` & ``numbers`` attributes.
        If ``title`` attribute is not included, 'Created with IODATA module' is used as ``title``.

    """
    print(data.natom, file=f)
    print(getattr(data, 'title', 'Created with IODATA module'), file=f)
    for i in range(data.natom):
        n = num2sym[data.numbers[i]]
        x, y, z = data.coordinates[i] / angstrom
        print(f'{n:2s} {x:15.10f} {y:15.10f} {z:15.10f}', file=f)

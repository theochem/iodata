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
"""Module for handling VASP POSCAR file format."""


from typing import Dict, TextIO

import numpy as np

from ..periodic import num2sym
from ..utils import angstrom, LineIterator
from .chgcar import _load_vasp_header


__all__ = ['load', 'dump']


patterns = ['POSCAR*']


def load(lit: LineIterator) -> Dict:
    """Load data from a VASP 5 POSCAR file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``title``, ``coordinates``, ``numbers`` & ``rvecs`` keys
        and their corresponding values.

    """
    # Load header
    title, rvecs, numbers, coordinates = _load_vasp_header(lit)
    return {
        'title': title,
        'coordinates': coordinates,
        'numbers': numbers,
        'rvecs': rvecs,
    }


def dump(f: TextIO, data: 'IOData'):
    """Write data into a VASP 5 POSCAR file format.

    Parameters
    ----------
    f
        A file to write to.
    data
        An IOData instance which must contain ``coordinates``, ``numbers``, ``rvecs`` &
        ``cell_frac`` attributes. It may contain ``title`` attribute.

    """
    print(getattr(data, 'title', 'Created with HORTON'), file=f)
    print('   1.00000000000000', file=f)

    # Write cell vectors, each row is one vector in angstrom:
    rvecs = data.rvecs
    for rvec in rvecs:
        r = rvec / angstrom
        print(f'{r[0]: 21.16f} {r[1]: 21.16f} {r[2]: 21.16f}', file=f)

    # Construct list of elements to make sure the coordinates get written
    # in this order. Heaviest elements are put furst.
    unumbers = sorted(np.unique(data.numbers))[::-1]
    print(' '.join(f'{num2sym[unumber]:5s}' for unumber in unumbers), file=f)
    print(' '.join(f'{(data.numbers == unumber).sum():5d}' for unumber in unumbers), file=f)
    print('Selective dynamics', file=f)
    print('Direct', file=f)

    # Write the coordinates
    for unumber in unumbers:
        indexes = (data.numbers == unumber).nonzero()[0]
        for index in indexes:
            row = np.dot(data.gvecs, data.coordinates[index])
            print(f'  {row[0]: 21.16f} {row[1]: 21.16f} {row[2]: 21.16f}   F   F   F', file=f)

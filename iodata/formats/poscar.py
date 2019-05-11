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
"""Module for handling VASP POSCAR file format."""


from typing import TextIO

import numpy as np

from ..periodic import num2sym
from ..utils import angstrom, LineIterator
from .chgcar import _load_vasp_header


__all__ = []


patterns = ['POSCAR*']


def load_one(lit: LineIterator) -> dict:
    """Load data from a VASP 5 POSCAR file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        Output dictionary containing ``title``, ``atcoords``, ``atnums`` & ``cellvecs`` keys
        and their corresponding values.

    """
    # Load header
    title, cellvecs, atnums, atcoords = _load_vasp_header(lit)
    return {
        'title': title,
        'atcoords': atcoords,
        'atnums': atnums,
        'cellvecs': cellvecs,
    }


def dump_one(f: TextIO, data: 'IOData'):
    """Write data into a VASP 5 POSCAR file format.

    Parameters
    ----------
    f
        A file to write to.
    data
        An IOData instance which must contain ``atcoords``, ``atnums``, ``cellvecs`` &
        ``cell_frac`` attributes. It may contain ``title`` attribute.

    """
    print(getattr(data, 'title', 'Created with HORTON'), file=f)
    print('   1.00000000000000', file=f)

    # Write cell vectors, each row is one vector in angstrom:
    cellvecs = data.cellvecs
    for rvec in cellvecs:
        r = rvec / angstrom
        print(f'{r[0]: 21.16f} {r[1]: 21.16f} {r[2]: 21.16f}', file=f)

    # Construct list of elements to make sure the coordinates get written
    # in this order. Heaviest elements are put furst.
    uatnums = sorted(np.unique(data.atnums))[::-1]
    print(' '.join(f'{num2sym[uatnum]:5s}' for uatnum in uatnums), file=f)
    print(' '.join(f'{(data.atnums == uatnum).sum():5d}' for uatnum in uatnums), file=f)
    print('Selective dynamics', file=f)
    print('Direct', file=f)

    # Write the coordinates
    for uatnum in uatnums:
        indexes = (data.atnums == uatnum).nonzero()[0]
        for index in indexes:
            row = np.dot(data.gvecs, data.atcoords[index])
            print(f'  {row[0]: 21.16f} {row[1]: 21.16f} {row[2]: 21.16f}   F   F   F', file=f)

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
"""VASP 5 POSCAR file format.

This format is used by `VASP 5.X <https://www.vasp.at/>`_ and
`VESTA <http://jp-minerals.org/vesta/en/>`_.
"""


from typing import TextIO

import numpy as np

from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..periodic import num2sym
from ..utils import angstrom, LineIterator
from .chgcar import _load_vasp_header


__all__ = []


PATTERNS = ['POSCAR*']


@document_load_one("VASP 5 POSCAR", ['atcoords', 'atnums', 'cellvecs', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Load header
    title, cellvecs, atnums, atcoords = _load_vasp_header(lit)
    return {
        'title': title,
        'atcoords': atcoords,
        'atnums': atnums,
        'cellvecs': cellvecs,
    }


@document_dump_one("VASP 5 POSCAR", ['atcoords', 'atnums', 'cellvecs'], ['title'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    print(data.title or 'Created with IOData', file=f)
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
    gvecs = np.linalg.inv(data.cellvecs).T
    for uatnum in uatnums:
        indexes = (data.atnums == uatnum).nonzero()[0]
        for index in indexes:
            row = np.dot(gvecs, data.atcoords[index])
            print(f'  {row[0]: 21.16f} {row[1]: 21.16f} {row[2]: 21.16f}   F   F   F', file=f)

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
"""MOL File Format.

This script handles either individual molecules or a molecular database
originating from V2000 compliant *.mol molecules.

More information on *.mol files can be found here
https://en.wikipedia.org/wiki/Chemical_table_file
"""

from typing import TextIO, Iterator

import numpy as np

from ..docstrings import (document_load_one, document_load_many)
from ..docstrings import (document_dump_one, document_dump_many)
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..utils import angstrom, LineIterator


__all__ = []

PATTERNS = ['*.mol']


@document_load_one("MOL", ['atcoords', 'atnums', 'bonds', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title = next(lit).strip()               # mol title row 1
    next(lit)                               # mol timestamp row 2
    next(lit)                               # mol blank line row 3
    words = next(lit).split()
    if words[-1].upper() != "V2000":
        lit.error("Only V2000 SDF files are supported.")
    natom = int(words[0])                   # number atoms col0 row 4
    nbond = int(words[1])                   # number bonds from col1 row4
    atcoords = np.empty((natom, 3), float)  # xyz coords  with shape 3
    atnums = np.empty(natom, int)           # each atom as iterated
    bonds = np.empty((nbond, 3), int)       # each bond as iterated
    for iatom in range(natom):              # iterate entire atom block
        words = next(lit).split()           # split each atom line
        atcoords[iatom, 0] = float(words[0]) * angstrom     # coord angstrom
        atcoords[iatom, 1] = float(words[1]) * angstrom     # coord angstrom
        atcoords[iatom, 2] = float(words[2]) * angstrom     # coord angstrom
        atnums[iatom] = sym2num.get(words[3].title())       # atomic symbol
    for ibond in range(nbond):              # iterate entire bond block
        words = next(lit).split()           # each element in bond line
        bonds[ibond, 0] = int(words[0])     # first atom for bond
        bonds[ibond, 1] = int(words[1])     # second atom for bond
        bonds[ibond, 2] = int(words[2])     # bond type (single,double,triple)
#        bonds[ibond, 3] = int(words[3])    # bond sterochemistry???
    while True:                             # iterate each commentary until EOF
        try:
            words = next(lit)               # skip each commentary line
        except StopIteration:               # what if file has wrong termination?
            lit.error("WARNING: Molecule specification did not end properly with M END !")
        if words == 'M  END\n':             # what if file has right termination?
            break                           # terminate while loop
    return{
        'title': title,
        'atcoords': atcoords,
        'atnums': atnums,
        'bonds': bonds}


@document_load_many("MOL", ['atcoords', 'atnums', 'bonds', 'title'])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    while True:
        try:
            yield load_one(lit)
        except StopIteration:
            return


@document_dump_one("MOL", ['atcoords', 'atnums', 'bonds', 'title'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    print(data.title or 'Created with IOData', file=f)    # row 1, title
    print('', file=f)                                     # row 2, empty
    print('', file=f)                                     # row 3, empty
    if data.bonds is None:                   # What if no bond block data?
        nbond = 0
    else:                                   # What if bond block data?
        nbond = len(data.bonds)             # num bonds is length of bonds array
    # populate line 4 with atom number and bond number from file
    print("{:3d}{:3d}  0     0  0  0  0  0  0999 V2000".format(data.natom, nbond), file=f)
    for beginatom in range(data.natom):         # populate entire atom block
        n = num2sym[data.atnums[beginatom]]
        x, y, z = data.atcoords[beginatom] / angstrom   # convert to arb units
        print(f'{x:10.4f}{y:10.4f}{z:10.4f} {n:<3s} 0  0  0  0  0  0  0  0  0  0  0  0', file=f)
    if data.bonds is not None:              # populate bond block
        for beginatom, endatom, bondtype in data.bonds:
            print('{:3d}{:3d}{:3d}  0  0  0  0'.format(beginatom, endatom, bondtype), file=f)
    print("M  END", file=f)


@document_dump_many("MOL", ['atcoords', 'atnums', 'bonds', 'title'])
def dump_many(f: TextIO, datas: IOData):
    """Do not edit this docstring. It will be overwritten."""
    for data in datas:
        dump_one(f, data)

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
"""SDF file format.

Usually, the different frames in a trajectory describe different geometries of the same
molecule, with atoms in the same order. The ``load_many`` and ``dump_many`` functions
below can also handle an SDF file with different molecules, e.g. a molecular database.

The SDF format is somewhat documented on the following page:
http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx

This format is one of the chemical table file formats:
https://en.wikipedia.org/wiki/Chemical_table_file
"""


from typing import TextIO, Iterator

import numpy as np

from ..docstrings import (document_load_one, document_load_many, document_dump_one,
                          document_dump_many)
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.sdf']


@document_load_one("SDF", ['atcoords', 'atnums', 'bonds', 'title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title = next(lit).strip()
    # The next two lines are general comments
    next(lit)
    next(lit)
    words = next(lit).split()
    natom = int(words[0])
    nbond = int(words[1])
    if words[-1].upper() != "V2000":
        lit.error("Only V2000 SDF files are supported.")
    atcoords = np.empty((natom, 3), float)
    atnums = np.empty(natom, int)
    for iatom in range(natom):
        words = next(lit).split()
        atcoords[iatom, 0] = float(words[0]) * angstrom
        atcoords[iatom, 1] = float(words[1]) * angstrom
        atcoords[iatom, 2] = float(words[2]) * angstrom
        atnums[iatom] = sym2num.get(words[3].title())
    bonds = np.empty((nbond, 3), int)
    for ibond in range(nbond):
        words = next(lit).split()
        bonds[ibond, 0] = int(words[0]) - 1
        bonds[ibond, 1] = int(words[1]) - 1
        # Bond types 1 to 8 (inclusive) are defined in the SDF format.
        # Anything outside that range is not modified, just not to lose any
        # information, but could be potentially meaningless.
        bonds[ibond, 2] = int(words[2])
    while True:
        try:
            words = next(lit)
        except StopIteration:
            lit.error("Molecule specification did not end properly with $$$$")
        if words == "$$$$\n":
            break
    return {
        'title': title,
        'atcoords': atcoords,
        'atnums': atnums,
        'bonds': bonds,
    }


@document_load_many("SDF", ['atcoords', 'atnums', 'bonds', 'title'])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # SDF files with more molecules are a simple concatenation of individual SDF files,'
    # making it travial to load many frames.
    while True:
        try:
            yield load_one(lit)
        except StopIteration:
            return


@document_dump_one("SDF", ['atcoords', 'atnums'], ['title', 'bonds'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    print(data.title or 'Created with IOData', file=f)
    print('', file=f)
    print('', file=f)
    nbond = 0 if data.bonds is None else len(data.bonds)
    print("{:3d}{:3d}  0     0  0  0  0  0  0999 V2000".format(data.natom, nbond), file=f)
    for iatom in range(data.natom):
        n = num2sym[data.atnums[iatom]]
        x, y, z = data.atcoords[iatom] / angstrom
        print(f'{x:10.4f}{y:10.4f}{z:10.4f} {n:<3s} 0  0  0  0  0  0  0  0  0  0  0  0', file=f)
    if data.bonds is not None:
        for iatom, jatom, bondtype in data.bonds:
            print('{:3d}{:3d}{:3d}  0  0  0  0'.format(
                iatom + 1, jatom + 1, bondtype
            ), file=f)
    print('M  END', file=f)
    print('$$$$', file=f)


@document_dump_many("SDF", ['atcoords', 'atnums'], ['title', 'bonds'])
def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

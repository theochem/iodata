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
"""MOL2 file format.

There are different formats of mol2 files. Here the compatibility with AMBER software
was the main objective to write out files with atomic charges used by antechamber.
"""


from typing import TextIO, Iterator, Tuple

import numpy as np

from ..docstrings import (document_load_one, document_load_many, document_dump_one,
                          document_dump_many)
from ..iodata import IOData, BOND_DTYPE
from ..periodic import sym2num, num2sym
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.mol2']


@document_load_one("MOL2", ['atcoords', 'atnums', 'atcharges', 'atffparams'], ['title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    molecule_found = False
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        if len(line) > 1:
            words = line.split()
            if words[0] == "@<TRIPOS>MOLECULE":
                # Found another molecule; go one line back and break
                if molecule_found:
                    lit.back(line)
                    break
                title = next(lit).strip()
                words = next(lit).split()
                natoms = int(words[0])
                nbonds = int(words[1])
            if words[0] == "@<TRIPOS>ATOM":
                atnums, atcoords, atchgs, attypes = _load_helper_atoms(lit, natoms)
                atcharges = {"mol2charges": atchgs}
                atffparams = {"attypes": attypes}
                result = {
                    'atcoords': atcoords,
                    'atnums': atnums,
                    'atcharges': atcharges,
                    'atffparams': atffparams,
                    'title': title
                }
                molecule_found = True
            if words[0] == "@<TRIPOS>BOND":
                bonds = _load_helper_bonds(lit, nbonds)
                result['bonds'] = bonds
    if molecule_found is False:
        raise lit.error("Molecule could not be read")
    return result


def _load_helper_atoms(lit: LineIterator, natoms: int)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Load element numbers, coordinates and atomic charges."""
    atnums = np.empty(natoms)
    atcoords = np.empty((natoms, 3))
    atchgs = np.empty(natoms)
    attypes = []
    for i in range(natoms):
        words = next(lit).split()
        # Check the first two characters of atom name and try
        # to convert to an element number both or only the first
        symbol = words[1][:2].title()
        atnum = sym2num.get(symbol, sym2num.get(symbol[0], None))
        if atnum is None:
            atnum = 0
            lit.warn(f'Can not convert {words[1][:2]} to elements')
        atnums[i] = atnum
        attypes.append(words[5])
        atcoords[i] = [float(words[2]), float(words[3]), float(words[4])]
        if len(words) == 9:
            atchgs[i] = float(words[8])
        else:
            atchgs[i] = 0.0000
    atcoords = atcoords * angstrom
    attypes = tuple(attypes)
    return atnums, atcoords, atchgs, attypes


def _load_helper_bonds(lit: LineIterator, nbonds: int) -> Tuple[np.ndarray]:
    """Load bond information."""
    bonds = np.empty(nbonds, dtype=BOND_DTYPE)
    for i in range(nbonds):
        words = next(lit).split()
        if words[3] == 'am':
            words[3] = 1.5
        # Substract one because of numbering starting at 0
        bonds[i] = int(words[1]) - 1, int(words[2]) - 1, float(words[3])
    return bonds


@document_load_many("MOL2", ['atcoords', 'atnums', 'atcharges', 'atffparams'], ['title'])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # MOL2 files with more molecules are a simple concatenation of individual MOL2 files,'
    # making it trivial to load many frames.
    while True:
        try:
            yield load_one(lit)
        except IOError:
            return


@document_dump_one("MOL2", ['atcoords', 'atnums'], ['atcharges', 'atffparams', 'title'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # The first six lines are reserved for comments
    print("# Mol2 file created with Iodata", file=f)
    print("\n\n\n\n\n", file=f)
    print("@<TRIPOS>MOLECULE", file=f)
    print(data.title or 'Created with IOData', file=f)
    if data.bonds is not None:
        bonds = len(data.bonds)
        print(f'{data.natom:5d} {bonds:6d} {0:6d} {0:6d}', file=f)
    else:
        print(f'{data.natom:5d} {0:6d} {0:6d} {0:6d}', file=f)
    print("@<TRIPOS>ATOM", file=f)
    atcharges = data.atcharges.get('mol2charges')
    attypes = data.atffparams.get('attypes')
    for i in range(data.natom):
        n = num2sym[data.atnums[i]]
        x, y, z = data.atcoords[i] / angstrom
        out1 = f'{i+1:7d} {n:2s} {x:15.4f} {y:9.4f} {z:9.4f} '
        atcharge = 0.0 if atcharges is None else atcharges[i]
        attype = n if attypes is None else attypes[i]
        out2 = f'{attype:6s} {1:4d} XXX {atcharge:14.4f}'
        print(out1 + out2, file=f)
    if data.bonds is not None:
        print("@<TRIPOS>BOND", file=f)
        for i, bond in enumerate(data.bonds):
            bondtype = 'am' if 1.4 < bond["bo"] < 1.6 else str(int(np.round(bond[2])))
            print(f'{i+1:6d} {bond["i"]+1:4d} {bond["j"]+1:4d} {bondtype:2s}',
                  file=f)


@document_dump_many("MOL2", ['atcoords', 'atnums', 'atcharges'], ['title'])
def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

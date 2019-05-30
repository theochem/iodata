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
from ..iodata import IOData
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
            if words[0] == "@<TRIPOS>ATOM":
                atnums, atcoords, atchgs, attypes = _load_helper_atoms(lit)
                atcharges = {"mol2charges": atchgs}
                atffparams = {"attypes": attypes}
                result = {
                    'atcoords': atcoords,
                    'atnums': atnums,
                    'atcharges': atcharges,
                    'atffparams': atffparams
                }
                if title is not None:
                    result['title'] = title
                molecule_found = True
    if molecule_found is False:
        raise lit.error("Molecule could not be read")
    return result


def _load_helper_atoms(lit: LineIterator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Load element numbers, coordinates and atomic charges."""
    atnums = []
    atcoords = []
    atchgs = []
    attypes = []
    for line in lit:
        words = line.split()
        if len(words) <= 6:
            break
        # Assume that the first character of atom type is the element
        try:
            atnums.append(sym2num.get(words[5][0].title()))
        except ValueError:
            print(f'Can not convert atom type {words[5][0]} to element')
        attypes.append(words[5])
        atcoords.append([float(words[2]), float(words[3]), float(words[4])])
        if len(words) == 9:
            atchgs.append(float(words[8]))
        else:
            atchgs.append(0.0000)
    atnums = np.array(atnums, int)
    atcoords = np.array(atcoords) * angstrom
    atchgs = np.array(atchgs)
    attypes = tuple(attypes)
    return atnums, atcoords, atchgs, attypes


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
    print("@<TRIPOS>ATOM", file=f)
    for i in range(data.natom):
        n = num2sym[data.atnums[i]]
        x, y, z = data.atcoords[i] / angstrom
        out1 = f'{i+1:7d} {n:2s} {x:15.4f} {y:9.4f} {z:9.4f} '
        atcharges = data.atcharges.get('mol2charges')
        attypes = data.atffparams.get('attypes')
        if atcharges is not None:
            charge = atcharges[i]
            if attypes is not None:
                attype = attypes[i]
                out2 = f'{attype:6s} {1:4d} XXX {charge:14.4f}'
            else:
                out2 = f'{n:6s} {1:4d} XXX {charge:14.4f}'
            print(out1 + out2, file=f)
        else:
            charge = 0.0000
            if attypes is not None:
                attype = attypes[i]
                out2 = f'{attype:6s} {1:4d} XXX {charge:14.4f}'
            else:
                out2 = f'{n:6s} {1:4d} XXX {charge:14.4f}'
            print(out1 + out2, file=f)


@document_dump_many("MOL2", ['atcoords', 'atnums', 'atcharges'], ['title'])
def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

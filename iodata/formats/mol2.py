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

from collections.abc import Iterator
from typing import TextIO
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from ..docstrings import (
    document_dump_many,
    document_dump_one,
    document_load_many,
    document_load_one,
)
from ..iodata import IOData
from ..periodic import bond2num, num2bond, num2sym, sym2num
from ..utils import LineIterator, LoadError, LoadWarning, angstrom

__all__ = []


PATTERNS = ["*.mol2"]


@document_load_one("MOL2", ["atcoords", "atnums", "atcharges", "atffparams"], ["title"])
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
                    "atcoords": atcoords,
                    "atnums": atnums,
                    "atcharges": atcharges,
                    "atffparams": atffparams,
                    "title": title,
                }
                molecule_found = True
            if words[0] == "@<TRIPOS>BOND":
                bonds = _load_helper_bonds(lit, nbonds)
                result["bonds"] = bonds
    if not molecule_found:
        raise LoadError("Molecule could not be read.", lit)
    return result


def _load_helper_atoms(
    lit: LineIterator, natoms: int
) -> tuple[NDArray[int], NDArray[float], NDArray[float], tuple]:
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
            warn(LoadWarning(f"Cannot interpret element symbol {words[1][:2]}", lit), stacklevel=2)
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


def _load_helper_bonds(lit: LineIterator, nbonds: int) -> NDArray[int]:
    """Load bond information.

    Each line in a bond definition has the following structure
    http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
    bond_index atom_1 atom_2 bond_type
    e.g.
    1 1 2 1
    This would be the first bond between atom 1 and atom 2 and a single bond
    """
    bonds = np.empty((nbonds, 3))
    for i in range(nbonds):
        words = next(lit).split()
        # Substract one because of numbering starting at 0
        bond = [
            int(words[1]) - 1,
            int(words[2]) - 1,
            # convert mol2 bond type to integer
            bond2num.get(words[3]),
        ]
        if bond[-1] is None:
            bond[-1] = bond2num["un"]
            warn(LoadWarning(f"Cannot interpret bond type {words[3]}", lit), stacklevel=2)
        bonds[i] = bond
    return bonds


@document_load_many("MOL2", ["atcoords", "atnums", "atcharges", "atffparams"], ["title"])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # MOL2 files with more molecules are a simple concatenation of individual MOL2 files,'
    # making it trivial to load many frames.
    try:
        while True:
            yield load_one(lit)
    except (StopIteration, LoadError):
        return


@document_dump_one("MOL2", ["atcoords", "atnums"], ["atcharges", "atffparams", "title"])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # The first six lines are reserved for comments
    print("# Mol2 file created with Iodata", file=f)
    print("\n\n\n\n\n", file=f)
    print("@<TRIPOS>MOLECULE", file=f)
    print(data.title or "Created with IOData", file=f)
    if data.bonds is not None:
        bonds = len(data.bonds)
        print(f"{data.natom:5d} {bonds:6d} {0:6d} {0:6d}", file=f)
    else:
        print(f"{data.natom:5d} {0:6d} {0:6d} {0:6d}", file=f)
    print("@<TRIPOS>ATOM", file=f)
    atcharges = data.atcharges.get("mol2charges")
    attypes = data.atffparams.get("attypes")
    for i in range(data.natom):
        n = num2sym[data.atnums[i]]
        x, y, z = data.atcoords[i] / angstrom
        out1 = f"{i+1:7d} {n:2s} {x:15.4f} {y:9.4f} {z:9.4f} "
        atcharge = 0.0 if atcharges is None else atcharges[i]
        attype = n if attypes is None else attypes[i]
        out2 = f"{attype:6s} {1:4d} XXX {atcharge:14.4f}"
        print(out1 + out2, file=f)
    if data.bonds is not None:
        print("@<TRIPOS>BOND", file=f)
        for i, bond in enumerate(data.bonds):
            bondtype = num2bond.get(bond[2], "un")
            print(f"{i+1:6d} {bond[0]+1:4d} {bond[1]+1:4d} {bondtype:2s}", file=f)


@document_dump_many("MOL2", ["atcoords", "atnums", "atcharges"], ["title"])
def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

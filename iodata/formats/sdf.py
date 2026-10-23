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

from collections.abc import Iterator
from typing import TextIO

import numpy as np

from ..docstrings import (
    document_dump_many,
    document_dump_one,
    document_load_many,
    document_load_one,
)
from ..iodata import IOData
from ..periodic import num2sym, sym2num
from ..utils import LineIterator, LoadError, angstrom

__all__ = ()


PATTERNS = ["*.sdf"]


def _parse_counts(line: str, lit: LineIterator) -> tuple[int, int]:
    """Parse the counts line, by column if possible."""
    if line.rsplit(maxsplit=1)[-1].upper() != "V2000":
        raise LoadError("Only V2000 SDF files are supported.", lit)
    try:
        return int(line[0:3]), int(line[3:6])
    except ValueError:
        words = line.split()
        return int(words[0]), int(words[1])


def _parse_atom(line: str) -> tuple[float, float, float, str]:
    """Parse an atom line, by column if possible."""
    symbol = line[31:34].strip()
    if symbol.title() in sym2num:
        try:
            return float(line[0:10]), float(line[10:20]), float(line[20:30]), symbol
        except ValueError:
            pass
    words = line.split()
    return float(words[0]), float(words[1]), float(words[2]), words[3]


def _parse_bond(line: str) -> tuple[int, int, int]:
    """Parse a bond line, by column if possible."""
    try:
        return int(line[0:3]), int(line[3:6]), int(line[6:9])
    except ValueError:
        words = line.split()
        return int(words[0]), int(words[1]), int(words[2])


@document_load_one("SDF", ["atcoords", "atnums", "bonds", "title"])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title = next(lit).strip()
    # The next two lines are general comments
    next(lit)
    next(lit)
    # The V2000 format has fixed-width fields, which may touch each other, e.g. for
    # molecules with 100 or more atoms. Each line is read by column first. Files that
    # do not follow the column layout are read by splitting at whitespace instead.
    natom, nbond = _parse_counts(next(lit), lit)
    atcoords = np.empty((natom, 3), float)
    atnums = np.empty(natom, int)
    for iatom in range(natom):
        x, y, z, symbol = _parse_atom(next(lit))
        atcoords[iatom] = np.array([x, y, z]) * angstrom
        atnums[iatom] = sym2num.get(symbol.title())
    bonds = np.empty((nbond, 3), int)
    for ibond in range(nbond):
        iatom0, iatom1, bondtype = _parse_bond(next(lit))
        bonds[ibond, 0] = iatom0 - 1
        bonds[ibond, 1] = iatom1 - 1
        # Bond types 1 to 8 (inclusive) are defined in the SDF format.
        # Anything outside that range is not modified, just not to lose any
        # information, but could be potentially meaningless.
        bonds[ibond, 2] = bondtype
    while True:
        try:
            words = next(lit)
        except StopIteration as exc:
            raise LoadError("Molecule specification did not end properly with $$$$.", lit) from exc
        if words == "$$$$\n":
            break
    return {
        "title": title,
        "atcoords": atcoords,
        "atnums": atnums,
        "bonds": bonds,
    }


@document_load_many("SDF", ["atcoords", "atnums", "bonds", "title"])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # SDF files with more molecules are a simple concatenation of individual SDF files,'
    # making it travial to load many frames.
    try:
        while True:
            yield load_one(lit)
    except StopIteration:
        return


@document_dump_one("SDF", ["atcoords", "atnums"], ["title", "bonds"])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    print(data.title or "Created with IOData", file=f)
    print("", file=f)
    print("", file=f)
    nbond = 0 if data.bonds is None else len(data.bonds)
    print(f"{data.natom:3d}{nbond:3d}  0     0  0  0  0  0  0999 V2000", file=f)
    for iatom in range(data.natom):
        n = num2sym[data.atnums[iatom]]
        x, y, z = data.atcoords[iatom] / angstrom
        print(f"{x:10.4f}{y:10.4f}{z:10.4f} {n:<3s} 0  0  0  0  0  0  0  0  0  0  0  0", file=f)
    if data.bonds is not None:
        for iatom, jatom, bondtype in data.bonds:
            print(f"{iatom + 1:3d}{jatom + 1:3d}{bondtype:3d}  0  0  0  0", file=f)
    print("M  END", file=f)
    print("$$$$", file=f)


@document_dump_many("SDF", ["atcoords", "atnums"], ["title", "bonds"])
def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

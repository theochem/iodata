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
"""PDB file format.

There are different formats of pdb files. The convention used here is the
last updated one and is described in this link:
http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
"""


from typing import TextIO, Iterator

import numpy as np

from ..docstrings import (document_load_one, document_load_many, document_dump_one,
                          document_dump_many)
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.pdb']


@document_load_one("PDB", ['atcoords', 'atnums', 'atffparams', 'extra'], ['title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Overview of ATOM records
    #     COLUMNS        DATA  TYPE    FIELD        DEFINITION
    # -------------------------------------------------------------------------------------
    #  1 -  6        Record name   "ATOM  "
    #  7 - 11        Integer       serial       Atom  serial number.
    # 13 - 16        Atom          name         Atom name.
    # 17             Character     altLoc       Alternate location indicator.
    # 18 - 20        Residue name  resName      Residue name.
    # 22             Character     chainID      Chain identifier.
    # 23 - 26        Integer       resSeq       Residue sequence number.
    # 27             AChar         iCode        Code for insertion of residues.
    # 31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
    # 39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
    # 47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
    # 55 - 60        Real(6.2)     occupancy    Occupancy.
    # 61 - 66        Real(6.2)     tempFactor   Temperature  factor.
    # 77 - 78        LString(2)    element      Element symbol, right-justified.
    # 79 - 80        LString(2)    charge       Charge  on the atom.
    atnums = []
    attypes = []
    restypes = []
    resnums = []
    coords = []
    occupancy = []
    bfactor = []
    molecule_found = False
    end_reached = False
    title = "PDB file from IOData"
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        # If the PDB file has a title replace it.
        if line.startswith("TITLE") or line.startswith("COMPND"):
            title = line[10:].rstrip()
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # get element symbol from position 77:78 in pdb format
            words = line[76:78].split()
            if not words:
                # If not present, guess it from position 13:16 (atom name)
                words = line[12:16].split()
            # assign atomic number
            symbol = words[0].title()
            atnums.append(sym2num.get(symbol, sym2num.get(symbol[0], None)))
            # atom name, residue name, chain id, & residue sequence number
            attypes.append(line[12:16].strip())
            restypes.append(line[17:20].strip())
            resnums.append(int(line[22:26]))
            # add x, y, and z
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            # get occupancy & temperature factor
            occupancy.append(float(line[54:60]))
            bfactor.append(float(line[60:66]))
            molecule_found = True
        if line.startswith("END") and molecule_found:
            end_reached = True
            break
    if molecule_found is False:
        lit.error("Molecule could not be read!")
    if not end_reached:
        lit.warn("The END is not found, but the parsed data is returned!")

    atffparams = {"attypes": np.array(attypes), "restypes": np.array(restypes),
                  "resnums": np.array(resnums)}
    extra = {"occupancy": np.array(occupancy), "bfactor": np.array(bfactor)}
    result = {
        'atcoords': np.array(coords) * angstrom,
        'atnums': np.array(atnums),
        'atffparams': atffparams,
        'title': title,
        'extra': extra
    }
    return result


@document_load_many("PDB", ['atcoords', 'atnums', 'atffparams', 'extra'], ['title'])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # PDB files with more molecules are a simple concatenation of individual PDB files,'
    # making it trivial to load many frames.
    while True:
        try:
            yield load_one(lit)
        except IOError:
            return


@document_dump_one("PDB", ['atcoords', 'atnums', 'extra'], ['atffparams', 'title'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    print(str("TITLE     " + data.title) or "TITLE      Created with IOData", file=f)
    attypes = data.atffparams.get('attypes', None)
    restypes = data.atffparams.get('restypes', None)
    resnums = data.atffparams.get('resnums', None)
    occupancy = data.extra.get('occupancy', None)
    bfactor = data.extra.get('bfactor', None)
    for i in range(data.natom):
        n = num2sym[data.atnums[i]]
        resnum = -1 if resnums is None else resnums[i]
        x, y, z = data.atcoords[i] / angstrom
        occ = 1.00 if occupancy is None else occupancy[i]
        b = 0.00 if bfactor is None else bfactor[i]
        attype = str(n + str(i + 1)) if attypes is None else attypes[i]
        restype = "XXX" if restypes is None else restypes[i]
        out1 = f'{i+1:>5d} {attype:<4s} {restype:3s} A{resnum:>4d}    '
        out2 = f'{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}{n:>12s}'
        print("ATOM  " + out1 + out2, file=f)
    print("END", file=f)


@document_dump_many("PDB", ['atcoords', 'atnums', 'extra'], ['atffparams', 'title'])
def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

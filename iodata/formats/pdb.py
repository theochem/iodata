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

from collections.abc import Iterator
from typing import TextIO
from warnings import warn

import numpy as np

from ..docstrings import (
    document_dump_many,
    document_dump_one,
    document_load_many,
    document_load_one,
)
from ..iodata import IOData
from ..periodic import bond2num, num2sym, sym2num
from ..utils import LineIterator, LoadError, LoadWarning, angstrom

__all__ = []


PATTERNS = ["*.pdb"]


def _parse_pdb_atom_line(line, lit):
    """Parse an ATOM or HETATM line from a PDB file.

    Parameters
    ----------
    line
        A string with a single ATOM or HETATM line.
    lit
        The line iterator which read the line, used for generating warnings when needed.

    Returns
    -------
    atnum
        Atomic number.
    atname
        The atom name.
    resname
        The residua name.
    chainid
        The chain ID.
    resnum
        The residue number.
    atcoord
        Cartesian coordinates of the atomic nucleus.
    occupancy
        The occupancy (usually from the XRD analysis).
    bfactor
        The temperature factor (usually from the XRD analysis).

    """
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

    # Get element symbol from position 77:78 in pdb format
    symbol = line[76:78].strip()
    if len(symbol) > 0:
        atnum = sym2num.get(symbol)
    else:
        # If not present, guess it from position 13:16 (atom name)
        atname = line[12:16].strip()
        atnum = sym2num.get(atname, sym2num.get(atname[:2].title(), sym2num.get(atname[0], None)))
        warn(
            LoadWarning("Using the atom name in the PDB file to guess the chemical element.", lit),
            stacklevel=2,
        )
    if atnum is None:
        atnum = 0
        warn(
            LoadWarning(
                f"Failed to determine the atomic number. atname='{atname}' symbol='{symbol}'", lit
            ),
            stacklevel=2,
        )

    # atom name, residue name, chain id, & residue sequence number
    atname = line[12:16].strip()
    resname = line[17:20].strip()
    chainid = line[21]
    resnum = int(line[22:26])
    # add x, y, and z
    atcoord = [
        float(line[30:38]) * angstrom,
        float(line[38:46]) * angstrom,
        float(line[46:54]) * angstrom,
    ]
    # get occupancies & temperature factor
    occupancy = float(line[54:60])
    bfactor = float(line[60:66])
    return atnum, atname, resname, chainid, resnum, atcoord, occupancy, bfactor


def _parse_pdb_conect_line(line):
    # Overview of CONECT records
    # COLUMNS       DATA  TYPE      FIELD        DEFINITION
    # -------------------------------------------------------------------------
    #  1 -  6       Record name    "CONECT"
    #  7 - 11       Integer        serial       Atom  serial number
    # 12 - 16       Integer        serial       Serial number of bonded atom
    # 17 - 21       Integer        serial       Serial number of bonded atom
    # 22 - 26       Integer        serial       Serial number of bonded atom
    # 27 - 31       Integer        serial       Serial number of bonded atom
    iatom0 = int(line[7:12]) - 1
    for ipos in 12, 17, 22, 27:
        serial_str = line[ipos : ipos + 5].strip()
        if serial_str != "":
            iatom1 = int(serial_str) - 1
            if iatom1 > iatom0:
                yield iatom0, iatom1


@document_load_one("PDB", ["atcoords", "atnums", "atffparams", "extra"], ["title", "bonds"])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title_lines = []
    compnd_lines = []
    atnums = []
    attypes = []
    restypes = []
    chainids = []
    resnums = []
    atcoords = []
    occupancies = []
    bfactors = []
    bonds = []
    molecule_found = False
    end_reached = False
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        # If the PDB file has a title, replace the default.
        if line.startswith("TITLE"):
            title_lines.append(line[10:].strip())
        if line.startswith("COMPND"):
            compnd_lines.append(line[10:].strip())
        if line.startswith(("ATOM", "HETATM")):
            (atnum, attype, restype, chainid, resnum, atcoord, occupancy, bfactor) = (
                _parse_pdb_atom_line(line, lit)
            )
            atnums.append(atnum)
            attypes.append(attype)
            restypes.append(restype)
            chainids.append(chainid)
            resnums.append(resnum)
            atcoords.append(atcoord)
            occupancies.append(occupancy)
            bfactors.append(bfactor)
            molecule_found = True
        if line.startswith("CONECT"):
            for iatom0, iatom1 in _parse_pdb_conect_line(line):
                bonds.append([iatom0, iatom1, bond2num["un"]])
        if line.startswith("END") and molecule_found:
            end_reached = True
            break
    if not molecule_found:
        raise LoadError("Molecule could not be read.", lit)
    if not end_reached:
        warn(
            LoadWarning("The END is not found, but the parsed data is returned.", lit), stacklevel=2
        )

    # Data related to force fields
    atffparams = {
        "attypes": np.array(attypes),
        "restypes": np.array(restypes),
        "resnums": np.array(resnums),
    }
    # Extra data
    extra = {
        "occupancies": np.array(occupancies),
        "bfactors": np.array(bfactors),
    }
    if len(compnd_lines) > 0:
        extra["compound"] = "\n".join(compnd_lines)
    # add chain id, if it wasn't all empty
    if not np.all(chainids == [" "] * len(chainids)):
        extra["chainids"] = np.array(chainids)
    # Set a useful title
    if len(title_lines) == 0:
        # Some files use COMPND instead of TITLE, in which case COMPND will be
        # used as title.
        if "compound" in extra:
            title = extra["compound"]
            del extra["compound"]
        else:
            title = "PDB file loaded by IOData"
    else:
        title = "\n".join(title_lines)
    result = {
        "atcoords": np.array(atcoords),
        "atnums": np.array(atnums),
        "atffparams": atffparams,
        "title": title,
        "extra": extra,
    }
    # assign bonds only if some were present
    if len(bonds) > 0:
        result["bonds"] = np.array(bonds)
    return result


@document_load_many("PDB", ["atcoords", "atnums", "atffparams", "extra"], ["title"])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # PDB files with more molecules are a simple concatenation of individual PDB files,'
    # making it trivial to load many frames.
    try:
        while True:
            yield load_one(lit)
    except (StopIteration, LoadError):
        return


def _dump_multiline_str(f: TextIO, key: str, value: str):
    r"""Write a multiline string in PDB format.

    Parameters
    ----------
    f
        A file object to write to.
    key
        The key used to prefix the multiline string, e.g. `"TITLE"`.
    value
        A (multiline) string, with multiple lines separated by `\n`.

    """
    prefix = key.ljust(10)
    for iline, line in enumerate(value.split("\n")):
        print(prefix + line, file=f)
        prefix = key + str(iline + 2).rjust(10 - len(key)) + " "


@document_dump_one("PDB", ["atcoords", "atnums", "extra"], ["atffparams", "title", "bonds"])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    _dump_multiline_str(f, "TITLE", data.title or "Created with IOData")
    if "compound" in data.extra:
        _dump_multiline_str(f, "COMPND", data.extra["compound"])
    # Prepare for ATOM lines.
    attypes = data.atffparams.get("attypes", None)
    restypes = data.atffparams.get("restypes", None)
    resnums = data.atffparams.get("resnums", None)
    occupancies = data.extra.get("occupancies", None)
    bfactors = data.extra.get("bfactors", None)
    chainids = data.extra.get("chainids", None)
    # Write ATOM lines.
    for i in range(data.natom):
        n = num2sym[data.atnums[i]]
        resnum = -1 if resnums is None else resnums[i]
        x, y, z = data.atcoords[i] / angstrom
        occ = 1.00 if occupancies is None else occupancies[i]
        b = 0.00 if bfactors is None else bfactors[i]
        attype = str(n + str(i + 1)) if attypes is None else attypes[i]
        restype = "XXX" if restypes is None else restypes[i]
        chain = " " if chainids is None else chainids[i]
        out1 = f"{i+1:>5d} {attype:<4s} {restype:3s} {chain:1s}{resnum:>4d}    "
        out2 = f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}{n:>12s}"
        print("ATOM  " + out1 + out2, file=f)
    if data.bonds is not None:
        # Prepare for CONECT lines.
        connections = [[] for iatom in range(data.natom)]
        for iatom0, iatom1 in data.bonds[:, :2]:
            connections[iatom0].append(iatom1)
            connections[iatom1].append(iatom0)
        # Write CONECT lines.
        for iatom0, iatoms1 in enumerate(connections):
            if len(iatoms1) > 0:
                # Write connection in groups of max 4
                for ichunk in range(len(iatoms1) // 4 + 1):
                    other_atoms_str = "".join(
                        f"{iatom1 + 1:5d}" for iatom1 in iatoms1[ichunk * 4 : ichunk * 4 + 4]
                    )
                    conect_line = f"CONECT{iatom0 + 1:5d}{other_atoms_str}"
                    print(conect_line, file=f)
    print("END", file=f)


@document_dump_many("PDB", ["atcoords", "atnums", "extra"], ["atffparams", "title"])
def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Do not edit this docstring. It will be overwritten."""
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

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
"""CHARMM crd file format.

CHARMM coordinate files contain information about the location of each atom in Cartesian space.
The format of the ASCII (CARD) CHARMM coordinate files is: Title line(s), number of atoms in file
and the coordinate lines (one for each atom in the file).

The coordinate lines contain specific information about each atom.
These have the following structure: Atom number (sequential), residue number
(specified relative to first residue in the PSF), residue name, atom type, x-coordinate,
y-coordinate, z-coordinate, segment identifier, residue identifier and a weighting array value.

"""


from typing import Tuple

import numpy as np

from ..docstrings import document_load_one

from ..utils import angstrom, amu, LineIterator

__all__ = []


PATTERNS = ['*.crd']


@document_load_one('CRD', ['atcoords', 'atffparams', 'atmasses', 'extra'], ['title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    # Read title section
    title = ''
    while True:
        try:
            line = next(lit)
        except StopIteration:
            lit.error("Title section of CRD has no ending marker (missing bare *).")
        # Get title from crd file.
        if line.startswith("*"):
            text = line[1:]
            if len(text.strip()) == 0:  # line with '*' only.
                break
            title += text
    # Read actual data
    data = _helper_read_crd(lit)
    resnums = np.array(data[0])
    resnames = np.array(data[1])
    attypes = np.array(data[2])
    atcoords = data[3]
    segid = np.array(data[4])
    resid = np.array(data[5])
    atmasses = np.array(data[6])
    atffparams = {
        'attypes': attypes,
        'resnames': resnames,
        'resnums': resnums
    }
    extra = {
        'segid': segid,
        'resid': resid,
    }
    result = {
        'atcoords': atcoords,
        'atffparams': atffparams,
        'atmasses': atmasses,
        'extra': extra,
        'title': title,
    }
    return result


def _helper_read_crd(lit: LineIterator) -> Tuple:
    """Read CHARMM crd file."""
    # Read the line for number of atoms.
    natom = next(lit)
    if natom is None or not natom.strip().isdigit():
        lit.error('The number of atoms must be an integer.')
    natom = int(natom)
    # Read the atom lines
    resnums = []
    resnames = []
    attypes = []
    pos = np.zeros((natom, 3), np.float32)
    segid = []
    resid = []
    atmasses = []
    for i in range(natom):
        line = next(lit)
        words = line.split()
        resnums.append(int(words[1]))
        resnames.append(words[2])
        attypes.append(words[3])
        pos[i, 0] = float(words[4])
        pos[i, 1] = float(words[5])
        pos[i, 2] = float(words[6])
        segid.append(words[7])
        resid.append(int(words[8]))
        atmasses.append(float(words[9]) * amu)
    pos *= angstrom
    return resnums, resnames, attypes, pos, segid, resid, atmasses

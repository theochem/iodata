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
"""CHARMm coordinate (crd) file format.

CHARMm coordinate files contain information about the location of each atom in Cartesian (3D) space.
The format of the ASCII (CARD) CHARMm coordinate files is:
    * Title line(s)
    * Number of atoms in file
    * Coordinate line (one for each atom in the file)

The coordinate lines contain specific information about each atom in the model:
    * Atom number (sequential)
    * Residue number (specified relative to first residue in the PSF)
    * Residue name
    * Atom type
    * X-coordinate
    * Y-coordinate
    * Z-coordinate
    * Segment identifier
    * Residue identifier
    * Weighting array value

"""


from typing import Tuple

import numpy as np

from ..docstrings import document_load_one

from ..utils import angstrom, LineIterator

__all__ = []


PATTERNS = ['*.crd']


@document_load_one('crd', ['atcoords', 'atffparams', 'extra'], ['title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title = ''
    title_end = False
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        # Get title from crd file.
        if '*' in line:
            text = line.split('*')[-1]
            if len(text) == 1:  # line with '*' only.
                title_end = True
            else:
                title += text
        if title_end:
            data = _helper_read_crd(lit)
            atnumbers = np.array(data[0])
            resnums = np.array(data[1])
            resnames = np.array(data[2])
            attypes = np.array(data[3])
            atcoords = data[4]
            segid = np.array(data[5])
            resid = np.array(data[6])
            weights = np.array(data[7])
            atffparams = {
                'attypes': attypes,
                'atnumbers': atnumbers,
                'resnames': resnames,
                'resnums': resnums
            }
            extra = {
                'segid': segid,
                'resid': resid,
                'weights': weights
            }
            result = {
                'atcoords': atcoords,
                'atffparams': atffparams,
                'extra': extra,
                'title': title,
            }
            break
    if title_end is False:
        raise lit.error('CHARMm crd file could not be read')
    return result


def _helper_read_crd(lit: LineIterator) -> Tuple:
    """Read CHARMm crd file."""
    # Read the line for number of atoms.
    natoms = next(lit)
    if natoms is not None:
        try:
            natoms = int(natoms)
        except TypeError:
            print('The number of atoms must be and integer.')
    # Read the atom lines
    atnumbers = []
    resnums = []
    resnames = []
    attypes = []
    pos = np.zeros((natoms, 3), np.float32)
    segid = []
    resid = []
    weights = []
    for i in range(natoms):
        line = next(lit)
        words = line.split()
        atnumbers.append(int(words[0]))
        resnums.append(int(words[1]))
        resnames.append(words[2])
        attypes.append(words[3])
        pos[i, 0] = float(words[4])
        pos[i, 1] = float(words[5])
        pos[i, 2] = float(words[6])
        segid.append(words[7])
        resid.append(int(words[8]))
        weights.append(float(words[9]))
    pos *= angstrom
    return atnumbers, resnums, resnames, attypes, pos, segid, resid, weights

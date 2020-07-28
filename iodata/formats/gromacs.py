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
"""gro file format.

Files with the gro file extension contain a molecular structure in Gromos87 format.
gro files can be used as trajectory by simply concatenating files.

"""


from typing import Tuple, Iterator

import numpy as np

from ..docstrings import (document_load_one, document_load_many)
from ..utils import nanometer, picosecond, LineIterator


__all__ = []


PATTERNS = ['*.gro']


@document_load_one('GRO', ['atcoords', 'atffparams', 'cellvecs', 'extra'], ['title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title = None
    frame_found = False
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        # Get title from gro file. Time field is optional.
        if 't=' in line:
            title = line.split(',')[0]
            frame_found = True
            lit.back(line)
            if frame_found:
                data = _helper_read_frame(lit)
                t = data[0]
                resnums = data[1]
                resnames = data[2]
                attypes = data[3]
                atnumbers = data[4]
                pos = data[5]
                vels = data[6]
                cell = data[7]
        if data:
            time = t
            resnums = np.array(resnums)
            resnames = np.array(resnames)
            attypes = np.array(attypes)
            atnumbers = np.array(atnumbers)
            atcoords = pos
            velocities = vels
            cellvecs = cell
            atffparams = {
                'attypes': attypes,
                'atnumbers': atnumbers,
                'resnames': resnames,
                'resnums': resnums
            }
            extra = {
                'time': time,
                'velocities': velocities
            }
            result = {
                'atcoords': atcoords,
                'atffparams': atffparams,
                'cellvecs': cellvecs,
                'extra': extra,
                'title': title,
            }
            break
    if frame_found is False:
        raise lit.error('gro file could not be read')
    return result


@document_load_many('GRO', ['atcoords', 'atffparams', 'cellvecs', 'extra'], ['title'])
def load_many(lit: LineIterator) -> Iterator[dict]:
    """Do not edit this docstring. It will be overwritten."""
    # gro files can be used as trajectory by simply concatenating files,
    # making it trivial to load many frames.
    while True:
        try:
            yield load_one(lit)
        except IOError:
            return


def _helper_read_frame(lit: LineIterator) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Read one frame."""
    # Read the first line, ignore the title and try to get the time.
    # Time field is optional.
    line = next(lit)
    pos = line.rfind('t=')
    if pos >= 0:
        time = float(line[pos + 2:]) * picosecond
    else:
        time = 0.0
    # Read the second line for number of atoms.
    natoms = next(lit)
    if natoms is not None:
        try:
            natoms = int(natoms)
        except TypeError:
            print('The number of atoms must be and integer.')
    # Read the atom lines
    resnums = []
    resnames = []
    attypes = []
    atnums = []
    pos = np.zeros((natoms, 3), np.float32)
    vel = np.zeros((natoms, 3), np.float32)
    for i in range(natoms):
        line = next(lit)
        resnums.append(int(line[:5]))
        resnames.append(line[5:10].split()[-1])
        attypes.append(line[10:15].split()[-1])
        atnums.append(int(line[15:20]))
        words = line[22:].split()
        pos[i, 0] = float(words[0])
        pos[i, 1] = float(words[1])
        pos[i, 2] = float(words[2])
        vel[i, 0] = float(words[3])
        vel[i, 1] = float(words[4])
        vel[i, 2] = float(words[5])
    pos *= nanometer  # atom coordinates are in nanometers
    vel *= nanometer / picosecond
    # Read the cell line
    cell = np.zeros((3, 3), np.float32)
    words = next(lit).split()
    if len(words) >= 3:
        cell[0, 0] = float(words[0])
        cell[1, 1] = float(words[1])
        cell[2, 2] = float(words[2])
    if len(words) == 9:
        cell[1, 0] = float(words[3])
        cell[2, 0] = float(words[4])
        cell[0, 1] = float(words[5])
        cell[2, 1] = float(words[6])
        cell[0, 2] = float(words[7])
        cell[1, 2] = float(words[8])
    cell *= nanometer
    return time, resnums, resnames, attypes, atnums, pos, vel, cell

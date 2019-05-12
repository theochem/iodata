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
"""Module for handling XYZ file format."""


from typing import TextIO, Iterator

import numpy as np

from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..utils import angstrom, LineIterator


__all__ = []


patterns = ['*.xyz']


def load_one(lit: LineIterator) -> dict:
    """Load molecular geometry from a XYZ file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out
        Output dictionary containing ``title`, ``atcoords`` & ``atnums`` keys
        and corresponding values.

    """
    size = int(next(lit))
    title = next(lit).strip()
    atcoords = np.empty((size, 3), float)
    atnums = np.empty(size, int)
    for i in range(size):
        words = next(lit).split()
        try:
            atnums[i] = sym2num[words[0].title()]
        except KeyError:
            atnums[i] = int(words[0])
        atcoords[i, 0] = float(words[1]) * angstrom
        atcoords[i, 1] = float(words[2]) * angstrom
        atcoords[i, 2] = float(words[3]) * angstrom
    return {
        'title': title,
        'atcoords': atcoords,
        'atnums': atnums
    }


def load_many(lit: LineIterator) -> Iterator[dict]:
    """Load an XYZ trajectory file.

    This function can also handle an XYZ with different molecules, e.g. a
    molecular database. Usually, the different frames in a trajectory describe
    different geometries of the same molecule, with atoms in the same order.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Yields
    ------
    output
        See XYZ load function above.

    """
    # XYZ Trajectory files are a simple concatenation of individual XYZ files,'
    # making it travial to load many frames.
    while True:
        try:
            yield load_one(lit)
        except StopIteration:
            return


def dump_one(f: TextIO, data: IOData):
    """Write molecular geometry into a XYZ file format.

    Parameters
    ----------
    f
        A file to write to.
    data
        An IOData instance which must contain ``atcoords`` & ``atnums`` attributes.
        If ``title`` attribute is not included, 'Created with IODATA module' is used as ``title``.

    """
    print(data.natom, file=f)
    print(data.title or 'Created with IOData', file=f)
    for i in range(data.natom):
        n = num2sym[data.atnums[i]]
        x, y, z = data.atcoords[i] / angstrom
        print(f'{n:2s} {x:15.10f} {y:15.10f} {z:15.10f}', file=f)


def dump_many(f: TextIO, datas: Iterator[IOData]):
    """Write an XYZ trajectory file.

    This will also work when the different data objects describe different
    molecules.

    Parameters
    ----------
    f
        A file to write to.
    datas
        An iterator of IOData instances, satisfying the requirements for the
        XYZ dump function above.

    """
    # Similar to load_many, this is relatively easy.
    for data in datas:
        dump_one(f, data)

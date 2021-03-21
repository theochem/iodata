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
"""Gaussian input format."""

import numpy as np

from ..docstrings import document_load_one
from ..periodic import sym2num
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ["*.com", "*.gjf"]


@document_load_one("Gaussian Input File", ['atcoords', 'atnums', 'title'], [])
def load_one(lit: LineIterator):
    """Do not edit this docstring. It will be overwritten."""
    line = next(lit)
    # check multiple-link 0 section starts with '%'
    while line.startswith(r'%'):
        line = next(lit)
    # check multiple-line route section
    data = {}

    route_line = ''
    while line.strip():
        route_line += (' ' + line.strip())
        line = next(lit)
    route_line = route_line[1:]

    line = next(lit)

    title_line = ''
    while line.strip():
        title_line += (' ' + line.strip())
        line = next(lit)
    title_line = title_line[1:]

    data['title'] = title_line

    # charge_spin_mult_line
    _ = next(lit)

    coord_line = next(lit)

    numbers = []
    coordinates = []
    while coord_line:
        contents = coord_line.strip().split()
        if not contents:
            break
        if len(contents) != 4:
            raise ValueError("No Cartesian Structure is detected")
        numbers.append(sym2num[contents[0]])
        coor = list(map(float, contents[1:]))
        coordinates.append(coor)
        coord_line = next(lit)
    data['atnums'] = np.array(numbers)
    data['atcoords'] = np.array(coordinates) * angstrom

    return data

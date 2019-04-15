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
"""Module for handling VASP LOCPOT file format."""


from typing import Dict

from ..utils import electronvolt, LineIterator
from .chgcar import _load_vasp_grid


__all__ = []


patterns = ['LOCPOT*']


def load(lit: LineIterator) -> Dict:
    """Load data from a VASP 5 LOCPOT file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Ouput dictionary containing ``title``, ``coordinates``, ``numbers``, ``rvecs``,
        ``grid`` & ``cube_data`` keys and corresponding values.

    """
    result = _load_vasp_grid(lit)
    # convert locpot to atomic units
    result['cube_data'] *= electronvolt
    return result

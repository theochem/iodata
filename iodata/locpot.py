# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
# pragma pylint: disable=invalid-name
"""Module for handling VASP POSCAR, CHGCAR and POTCAR file formats."""


from typing import Dict

from .utils import electronvolt
from .chgcar import _load_vasp_grid


__all__ = ['load_locpot']


def load_locpot(filename: str) -> Dict:
    """Load data from a VASP 5 LOCPOT file format.

    Parameters
    ----------
    filename
        The VASP 5 LOCPOT filename.

    Returns
    -------
    out : dict
        Ouput dictionary containing ``title``, ``coordinates``, ``numbers``, ``rvecs``,
        ``grid`` & ``cube_data`` keys and corresponding values.

    """
    result = _load_vasp_grid(filename)
    # convert locpot to atomic units
    result['cube_data'] *= electronvolt
    return result

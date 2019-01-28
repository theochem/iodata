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
# pragma pylint: disable=invalid-name,no-member
"""Test iodata.chgcar module."""


import numpy as np

from .. utils import angstrom, volume
from .. iodata import IOData
from .. chgcar import _unravel_counter
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_unravel_counter():
    assert _unravel_counter(0, [3, 3, 3]) == [0, 0, 0]
    assert _unravel_counter(0, [2, 4, 3]) == [0, 0, 0]
    assert _unravel_counter(1, [2, 4, 3]) == [1, 0, 0]
    assert _unravel_counter(2, [2, 4, 3]) == [0, 1, 0]
    assert _unravel_counter(3, [2, 4, 3]) == [1, 1, 0]
    assert _unravel_counter(8, [2, 4, 3]) == [0, 0, 1]
    assert _unravel_counter(9, [2, 4, 3]) == [1, 0, 1]
    assert _unravel_counter(11, [2, 4, 3]) == [1, 1, 1]
    assert _unravel_counter(24, [2, 4, 3]) == [0, 0, 0]


def test_load_chgcar_oxygen():
    with path('iodata.test.data', 'CHGCAR.oxygen') as fn:
        mol = IOData.from_file(str(fn))
    assert (mol.numbers == [8]).all()
    assert abs(volume(mol.rvecs) - (10 * angstrom) ** 3) < 1e-10
    ugrid = mol.grid
    assert len(ugrid['shape']) == 3
    assert (ugrid['shape'] == 2).all()
    assert abs(ugrid['grid_rvecs'] - mol.rvecs / 2).max() < 1e-10
    assert abs(ugrid['origin']).max() < 1e-10
    d = mol.cube_data
    assert abs(d[0, 0, 0] - 0.78406017013E+04 / volume(mol.rvecs)) < 1e-10
    assert abs(d[-1, -1, -1] - 0.10024522914E+04 / volume(mol.rvecs)) < 1e-10
    assert abs(d[1, 0, 0] - 0.76183317989E+04 / volume(mol.rvecs)) < 1e-10


def test_load_chgcar_water():
    with path('iodata.test.data', 'CHGCAR.water') as fn:
        mol = IOData.from_file(str(fn))
    assert mol.title == 'unknown system'
    assert (mol.numbers == [8, 1, 1]).all()
    coords = np.array([0.074983 * 15 + 0.903122 * 1, 0.903122 * 15, 0.000000])
    assert abs(mol.coordinates[1] - coords).max() < 1e-7
    assert abs(volume(mol.rvecs) - 15 ** 3) < 1e-4
    ugrid = mol.grid
    assert len(ugrid['shape']) == 3
    assert (ugrid['shape'] == 3).all()
    assert abs(ugrid['grid_rvecs'] - mol.rvecs / 3).max() < 1e-10
    assert abs(ugrid['origin']).max() < 1e-10

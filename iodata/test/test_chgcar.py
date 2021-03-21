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
# pylint: disable=unsubscriptable-object
"""Test iodata.formats.chgcar module."""

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one
from ..utils import angstrom, volume

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_chgcar_oxygen():
    with path('iodata.test.data', 'CHGCAR.oxygen') as fn:
        mol = load_one(str(fn))
    assert_equal(mol.atnums, 8)
    assert_allclose(volume(mol.cellvecs), (10 * angstrom) ** 3, atol=1.e-10)
    assert_equal(mol.cube.shape, [2, 2, 2])
    assert abs(mol.cube.origin).max() < 1e-10
    assert_allclose(mol.cube.axes, mol.cellvecs / 2, atol=1.e-10)
    d = mol.cube.data
    assert_allclose(d[0, 0, 0], 0.78406017013E+04 / volume(mol.cellvecs), atol=1.e-10)
    assert_allclose(d[-1, -1, -1], 0.10024522914E+04 / volume(mol.cellvecs), atol=1.e-10)
    assert_allclose(d[1, 0, 0], 0.76183317989E+04 / volume(mol.cellvecs), atol=1.e-10)


def test_load_chgcar_water():
    with path('iodata.test.data', 'CHGCAR.water') as fn:
        mol = load_one(str(fn))
    assert mol.title == 'unknown system'
    assert_equal(mol.atnums, [8, 1, 1])
    coords = np.array([0.074983 * 15 + 0.903122 * 1, 0.903122 * 15, 0.000000])
    assert_allclose(mol.atcoords[1], coords, atol=1.e-7)
    assert_allclose(volume(mol.cellvecs), 15 ** 3, atol=1.e-4)
    assert_equal(len(mol.cube.shape), 3)
    assert_equal(mol.cube.shape, (3, 3, 3))
    assert_allclose(mol.cube.axes, mol.cellvecs / 3, atol=1.e-10)
    assert abs(mol.cube.origin).max() < 1e-10

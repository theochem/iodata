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
"""Test iodata.formats.locpot module."""

from numpy.testing import assert_equal, assert_allclose

from ..api import load_one
from ..utils import angstrom, electronvolt, volume
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_locpot_oxygen():
    with path('iodata.test.data', 'LOCPOT.oxygen') as fn:
        mol = load_one(str(fn))
    assert mol.title == 'O atom in a box'
    assert_equal(mol.atnums[0], 8)
    assert_allclose(volume(mol.cellvecs), (10 * angstrom) ** 3, atol=1.e-10)
    assert_equal(len(mol.cube.shape), 3)
    assert_equal(mol.cube.shape, [1, 4, 2])
    assert abs(mol.cube.origin).max() < 1e-10
    d = mol.cube.data
    assert_allclose(d[0, 0, 0] / electronvolt, 0.35046350435E+01, 1.e-10)
    assert_allclose(d[0, 1, 0] / electronvolt, 0.213732132354E+01, 1.e-10)
    assert_allclose(d[0, 2, 0] / electronvolt, -.65465465497E+01, 1.e-10)
    assert_allclose(d[0, 2, 1] / electronvolt, -.546876467887E+01, 1.e-10)

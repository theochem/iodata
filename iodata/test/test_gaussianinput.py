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
"""Test iodata.formats.gaussianinput module."""

import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises

from ..api import load_one
from ..utils import angstrom
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_water_com():
    # test .com with Link 0 section
    with path('iodata.test.data', 'water.com') as fn_xyz:
        mol = load_one(str(fn_xyz))
    check_water(mol, 'water')


def test_load_water_gjf():
    # test .com without Link 0 section
    with path('iodata.test.data', 'water.gjf') as fn_xyz:
        mol = load_one(str(fn_xyz))
    check_water(mol, 'water')


def test_load_multi_link():
    # test .com with multiple #link 0 contents
    with path('iodata.test.data', 'water_multi_link.com') as fn_xyz:
        mol = load_one(str(fn_xyz))
    check_water(mol, 'water')


def test_load_multi_route():
    # test .com with multiple route contents
    with path('iodata.test.data', 'water_multi_route.com') as fn_xyz:
        mol = load_one(str(fn_xyz))
    check_water(mol, 'water')


def test_load_multi_title():
    # test .com with multiple title and concatenate
    with path('iodata.test.data', 'water_multi_title.com') as fn_xyz:
        mol = load_one(str(fn_xyz))
    check_water(mol, 'water water')


def test_load_error():
    # test error raises when loading .com with z-matrix
    with assert_raises(ValueError):
        with path('iodata.test.data', 'water_z.com') as fn_xyz:
            load_one(str(fn_xyz))


def check_water(mol, title):
    """Test water molecule attributes."""
    assert mol.title == title
    assert_equal(mol.atnums, [1, 8, 1])
    # check bond length
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[1]) / angstrom, 0.960, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.atcoords[2] - mol.atcoords[1]) / angstrom, 0.960, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[2]) / angstrom, 1.568, atol=1.e-3)

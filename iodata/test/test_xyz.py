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
# pylint: disable=no-member
"""Test iodata.xyz module."""

import os

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..iodata import load_one, load_many, dump_one, dump_many
from ..utils import angstrom
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_water_number():
    # test xyz with atomic numbers
    with path('iodata.test.data', 'water_number.xyz') as fn_xyz:
        mol = load_one(str(fn_xyz))
    check_water(mol)


def test_load_water_element():
    # test xyz file with atomic symbols
    with path('iodata.test.data', 'water_element.xyz') as fn_xyz:
        mol = load_one(str(fn_xyz))
    check_water(mol)


def check_water(mol):
    """Test some things on a water file."""
    assert mol.title == 'Water'
    assert_equal(mol.numbers, [1, 8, 1])
    # check bond length
    print(np.linalg.norm(mol.coordinates[0] - mol.coordinates[2]) / angstrom)
    assert_allclose(np.linalg.norm(
        mol.coordinates[0] - mol.coordinates[1]) / angstrom, 0.960, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.coordinates[2] - mol.coordinates[1]) / angstrom, 0.960, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.coordinates[0] - mol.coordinates[2]) / angstrom, 1.568, atol=1.e-3)


def check_load_dump_consistency(tmpdir, fn):
    """Check if dumping and loading an XYZ file results in the same data."""
    mol0 = load_one(str(fn))
    # write xyz file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test.xyz')
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    # check two xyz files
    assert mol0.title == mol1.title
    assert_equal(mol0.numbers, mol1.numbers)
    assert_allclose(mol0.coordinates, mol1.coordinates, atol=1.e-5)


def test_load_dump_consistency(tmpdir):
    with path('iodata.test.data', 'ch3_hf_sto3g.fchk') as fn_fchk:
        check_load_dump_consistency(tmpdir, fn_fchk)


def test_dump_xyz_water_element(tmpdir):
    with path('iodata.test.data', 'water_element.xyz') as fn_xyz:
        check_load_dump_consistency(tmpdir, fn_xyz)


def test_dump_xyz_water_number(tmpdir):
    with path('iodata.test.data', 'water_number.xyz') as fn_xyz:
        check_load_dump_consistency(tmpdir, fn_xyz)


def test_load_many():
    with path('iodata.test.data', 'water_trajectory.xyz') as fn_xyz:
        mols = list(load_many(str(fn_xyz)))
    assert len(mols) == 5
    for imol, mol in enumerate(mols):
        assert mol.title == "Frame {}".format(imol)
        assert_equal(mol.numbers, [8, 1, 1])
        assert mol.coordinates.shape == (3, 3)
    assert_allclose(mols[0].coordinates[2] / angstrom, [2.864329, 0.114369, 3.3635])
    assert_allclose(mols[2].coordinates[0] / angstrom, [-0.232964, -0.789588, -3.247615])
    assert_allclose(mols[-1].coordinates[1] / angstrom, [-2.123423, -3.355326, -3.353739])


def test_load_dump_many_consistency(tmpdir):
    with path('iodata.test.data', 'water_trajectory.xyz') as fn_xyz:
        mols0 = list(load_many(str(fn_xyz)))
    # write xyz file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test.xyz')
    dump_many(mols0, fn_tmp)
    mols1 = list(load_many(fn_tmp))
    assert len(mols0) == len(mols1)
    for mol0, mol1 in zip(mols0, mols1):
        assert mol0.title == mol1.title
        assert_equal(mol0.numbers, mol1.numbers)
        assert_allclose(mol0.coordinates, mol1.coordinates, atol=1.e-5)

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
"""Test iodata.formats.sdf module."""

import os

import pytest
from numpy.testing import assert_equal, assert_allclose

from .common import truncated_file
from ..api import load_one, load_many, dump_one, dump_many
from ..utils import angstrom, FileFormatError

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_sdf_load_one_example():
    # test sdf one structure
    with path('iodata.test.data', 'example.sdf') as fn_sdf:
        mol = load_one(str(fn_sdf))
    check_example(mol)


def test_sdf_load_one_formamide():
    # test sdf one structure
    with path('iodata.test.data', 'formamide.sdf') as fn_sdf:
        mol = load_one(str(fn_sdf))
    assert mol.title == "713"
    assert mol.natom == 6
    assert len(mol.bonds) == 5
    assert_equal(mol.atnums, [8, 7, 6, 1, 1, 1])
    assert_equal(mol.bonds, [[0, 2, 2], [1, 2, 1], [1, 3, 1], [1, 4, 1], [2, 5, 1]])


def test_sdf_formaterror(tmpdir):
    # test if sdf file has the wrong ending without $$$$
    with path('iodata.test.data', 'example.sdf') as fn_test:
        with truncated_file(fn_test, 36, 0, tmpdir) as fn:
            with pytest.raises(IOError):
                load_one(str(fn))


def check_example(mol):
    """Test some things on example file."""
    assert mol.title == '24978498'
    assert mol.natom == 16
    assert len(mol.bonds) == 15
    assert_equal(mol.atnums, [16, 8, 8, 8, 8, 7, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1])
    # check coordinates
    atcoords_ang = mol.atcoords / angstrom
    assert_allclose(atcoords_ang[0], [2.8660, -0.4400, 0.0000])
    assert_allclose(atcoords_ang[1], [5.4641, 1.0600, 0.0000])
    assert_allclose(atcoords_ang[14], [6.0010, 1.3700, 0.0000])
    assert_allclose(atcoords_ang[15], [2.0000, -2.5600, 0.0000])
    assert_equal(mol.bonds[0], [0, 3, 1])
    assert_equal(mol.bonds[4], [2, 8, 2])
    assert_equal(mol.bonds[14], [7, 11, 1])


def check_load_dump_consistency(tmpdir, fn):
    """Check if dumping and loading an SDF file results in the same data."""
    mol0 = load_one(str(fn))
    # write sdf file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test.sdf')
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    # check two sdf files
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-5)
    assert_equal(mol0.bonds, mol1.bonds)


def test_load_dump_consistency(tmpdir):
    with path('iodata.test.data', 'example.sdf') as fn_sdf:
        check_load_dump_consistency(tmpdir, fn_sdf)
    with path('iodata.test.data', 'formamide.sdf') as fn_sdf:
        check_load_dump_consistency(tmpdir, fn_sdf)
    # The benzene mol2 file has aromatic bonds, which are less common in SDF files.
    with path('iodata.test.data', 'benzene.mol2') as fn_sdf:
        check_load_dump_consistency(tmpdir, fn_sdf)


def test_load_many():
    with path('iodata.test.data', 'example.sdf') as fn_sdf:
        mols = list(load_many(str(fn_sdf)))
    assert len(mols) == 2
    check_example(mols[0])
    assert mols[1].title == '24978481'
    assert_equal(mols[1].natom, 21)
    assert_allclose(mols[0].atcoords[0] / angstrom, [2.8660, -0.4400, 0.0000])
    assert_allclose(mols[1].atcoords[1] / angstrom, [1.4030, 1.4030, 0.0000])


def test_load_dump_many_consistency(tmpdir):
    with path('iodata.test.data', 'example.sdf') as fn_sdf:
        mols0 = list(load_many(str(fn_sdf)))
    # write sdf file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test')
    dump_many(mols0, fn_tmp, fmt='sdf')
    mols1 = list(load_many(fn_tmp, fmt='sdf'))
    assert len(mols0) == len(mols1)
    for mol0, mol1 in zip(mols0, mols1):
        assert mol0.title == mol1.title
        assert_equal(mol0.atnums, mol1.atnums)
        assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-5)
        assert_equal(mol0.bonds, mol1.bonds)


def test_v2000_check():
    with path('iodata.test.data', 'molv3000.sdf') as fn_sdf:
        with pytest.raises(FileFormatError):
            load_one(fn_sdf)

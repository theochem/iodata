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
"""Test iodata.formats.mol module."""

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


def test_mol_load_one_water():
    # test mol one structure
    with path('iodata.test.data', 'water.mol') as fn_mol:
        mol = load_one(str(fn_mol))
    assert mol.title == 'H2O'
    assert mol.natom == 3
    assert len(mol.bonds) == 2
    assert_equal(mol.atnums, [8, 1, 1])
    # check coordinates
    atcoords_ang = mol.atcoords / angstrom
    assert_allclose(atcoords_ang[0], [-0.0000, -0.0589, -0.0000])
    assert_allclose(atcoords_ang[1], [-0.8110, 0.4677, 0.0000])
    assert_allclose(atcoords_ang[2], [0.8110, 0.4677, 0.0000])
    assert_equal(mol.bonds[0], [0, 1, 1])
    assert_equal(mol.bonds[1], [0, 2, 1])


def test_mol_load_one_benzoicacid():
    # test mol one structure
    with path('iodata.test.data', 'benzoicacid.mol') as fn_mol:
        mol = load_one(str(fn_mol))
    assert mol.title == 'C7H6O2'
    assert mol.natom == 15
    assert len(mol.bonds) == 15
    assert_equal(mol.atnums, [6, 6, 6, 6, 6, 6, 6, 8, 8, 1, 1, 1, 1, 1, 1])
    atcoords_ang = mol.atcoords / angstrom
    # test atoms
    assert_allclose(atcoords_ang[0], [1.6471, 0.0768, -0.0012])
    assert_allclose(atcoords_ang[1], [0.1706, 0.0297, -0.0002])
    assert_allclose(atcoords_ang[2], [-0.5689, 1.2141, 0.0004])
    assert_allclose(atcoords_ang[3], [-1.9473, 1.1628, 0.0009])
    assert_allclose(atcoords_ang[4], [-2.5974, -0.0586, 0.0013])
    assert_allclose(atcoords_ang[5], [-1.8708, -1.2362, 0.0011])
    assert_allclose(atcoords_ang[6], [-0.4918, -1.1994, -0.0057])
    assert_allclose(atcoords_ang[7], [2.2206, 1.1476, -0.0015])
    assert_allclose(atcoords_ang[8], [2.3575, -1.0677, 0.0038])
    assert_allclose(atcoords_ang[9], [-0.0627, 2.1682, 0.0009])
    assert_allclose(atcoords_ang[10], [-2.5205, 2.0782, 0.0014])
    assert_allclose(atcoords_ang[11], [-3.6768, -0.0931, 0.0020])
    assert_allclose(atcoords_ang[12], [-2.3844, -2.1862, 0.0021])
    assert_allclose(atcoords_ang[13], [0.0741, -2.1193, -0.0061])
    assert_allclose(atcoords_ang[14], [3.3211, -0.9864, 0.0029])
    # test bonds
    assert_equal(mol.bonds[0], [0, 1, 1])
    assert_equal(mol.bonds[1], [1, 2, 2])
    assert_equal(mol.bonds[2], [2, 3, 1])
    assert_equal(mol.bonds[3], [3, 4, 2])
    assert_equal(mol.bonds[4], [4, 5, 1])
    assert_equal(mol.bonds[5], [5, 6, 2])
    assert_equal(mol.bonds[6], [1, 6, 1])
    assert_equal(mol.bonds[7], [0, 7, 2])
    assert_equal(mol.bonds[8], [0, 8, 1])
    assert_equal(mol.bonds[9], [2, 9, 1])
    assert_equal(mol.bonds[10], [3, 10, 1])
    assert_equal(mol.bonds[11], [4, 11, 1])
    assert_equal(mol.bonds[12], [5, 12, 1])
    assert_equal(mol.bonds[13], [6, 13, 1])
    assert_equal(mol.bonds[14], [8, 14, 1])


def test_mol_formaterror(tmpdir):
    # test mol file if it has an ending other than 'M  END'
    with path('iodata.test.data', 'benzoicacid.mol') as fn_test:
        with truncated_file(fn_test, 15, 0, tmpdir) as fn:
            with pytest.raises(IOError):
                load_one(str(fn))


def check_load_dump_consistency(tmpdir, fn):
    """Check if dumping and loading an MOL file results in the same data."""
    mol0 = load_one(str(fn))
    fn_tmp = os.path.join(tmpdir, 'test.mol')
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-5)
    assert_equal(mol0.bonds, mol1.bonds)


def test_dump_consistency(tmpdir):
    with path('iodata.test.data', 'water.mol') as fn_mol:
        check_load_dump_consistency(tmpdir, fn_mol)
    with path('iodata.test.data', 'benzoicacid.mol') as fn_mol:
        check_load_dump_consistency(tmpdir, fn_mol)
    with path('iodata.test.data', 'benzene.mol2') as fn_mol:
        check_load_dump_consistency(tmpdir, fn_mol)


def test_load_many():
    with path('iodata.test.data', 'database.mol') as fn_mol:
        mols = list(load_many(str(fn_mol)))
    assert len(mols) == 2
    assert mols[1].title == 'H2O'
    assert mols[0].title == 'H2O'
    assert_equal(mols[0].bonds[0], [0, 1, 1])
    assert_equal(mols[1].bonds[0], [0, 1, 1])
    assert_equal(mols[0].bonds[1], [0, 2, 1])
    assert_equal(mols[1].bonds[1], [0, 2, 1])
    assert_equal(mols[0].natom, mols[1].natom)
    assert_equal(mols[1].natom, 3)
    assert_allclose(mols[0].atcoords[0] / angstrom, [-0.0000, -0.0589, -0.0000])
    assert_allclose(mols[1].atcoords[0] / angstrom, [-0.0000, -0.0589, -0.0000])
    assert_allclose(mols[0].atcoords[1] / angstrom, [-0.8110, 0.4677, 0.0000])
    assert_allclose(mols[1].atcoords[1] / angstrom, [-0.8110, 0.4677, 0.0000])


def test_load_dump_many_consistency(tmpdir):
    with path('iodata.test.data', 'water.mol') as fn_mol:
        mols0 = list(load_many(str(fn_mol)))
    fn_tmp = os.path.join(tmpdir, 'test')
    dump_many(mols0, fn_tmp, fmt='mol')
    mols1 = list(load_many(fn_tmp, fmt='mol'))
    assert len(mols0) == len(mols1)
    for mol0, mol1 in zip(mols0, mols1):
        assert mol0.title == mol1.title
        assert_equal(mol0.atnums, mol1.atnums)
        assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-5)
        assert_equal(mol0.bonds, mol1.bonds)


def test_v2000_check():
    with path('iodata.test.data', 'waterv3000.mol') as fn_mol:
        with pytest.raises(FileFormatError):
            load_one(fn_mol)

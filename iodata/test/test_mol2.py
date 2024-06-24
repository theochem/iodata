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
"""Test iodata.formats.mol2 module."""

import os
from importlib.resources import as_file, files

import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import dump_many, dump_one, load_many, load_one
from ..periodic import bond2num
from ..utils import LoadError, LoadWarning, angstrom
from .common import truncated_file


def test_mol2_load_one():
    # test mol2 one structure
    with as_file(files("iodata.test.data").joinpath("caffeine.mol2")) as fn_mol:
        mol = load_one(str(fn_mol))
    check_example(mol)


def test_mol2_formaterror(tmpdir):
    # test if mol2 file has the wrong ending
    with (
        as_file(files("iodata.test.data").joinpath("caffeine.mol2")) as fn_test,
        truncated_file(fn_test, 2, 0, tmpdir) as fn,
        pytest.raises(LoadError),
    ):
        load_one(str(fn))


def test_mol2_symbol():
    # test mol2 files with element symbols with two characters
    with as_file(files("iodata.test.data").joinpath("silioh3.mol2")) as fn_mol:
        mol = load_one(str(fn_mol))
    assert_equal(mol.atnums, [14, 3, 8, 1, 8, 1, 8, 1])


def test_bondtypes_benzene():
    with as_file(files("iodata.test.data").joinpath("benzene.mol2")) as fn_test:
        mol = load_one(str(fn_test))
    assert_equal(mol.bonds[:, 2], [bond2num["ar"]] * 6 + [bond2num["1"]] * 6)


def check_example(mol):
    """Test some things on example file."""
    assert mol.title == "ZINC00001084"
    assert_equal(mol.natom, 24)
    assert_equal(
        mol.atnums, [6, 7, 6, 7, 6, 6, 6, 8, 7, 6, 8, 7, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    assert mol.atffparams["attypes"][0] == "C.3"
    # check coordinates
    atcoords_ang = mol.atcoords / angstrom
    assert_allclose(atcoords_ang[0], [-0.0178, 1.4608, 0.0101])
    assert_allclose(atcoords_ang[1], [0.0021, -0.0041, 0.0020])
    assert_allclose(atcoords_ang[22], [0.5971, -2.2951, 5.2627])
    assert_allclose(atcoords_ang[23], [0.5705, -0.5340, 5.0055])
    assert_allclose(mol.atcharges["mol2charges"][0], 0.0684)
    assert_allclose(mol.atcharges["mol2charges"][23], 0.0949)
    bonds = mol.bonds
    assert len(bonds) == 25
    assert_equal(bonds[0], [0, 1, bond2num["1"]])
    assert_equal(bonds[6], [2, 3, bond2num["2"]])
    assert_equal(bonds[13], [6, 8, bond2num["am"]])
    assert_equal(bonds[24], [13, 23, bond2num["1"]])


def check_load_dump_consistency(tmpdir, fn):
    """Check if dumping and loading an MOL2 file results in the same data."""
    mol0 = load_one(str(fn))
    # write mol2 file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, "test.mol2")
    dump_one(mol0, fn_tmp, fmt="mol2")
    mol1 = load_one(fn_tmp)
    # check two mol2 files
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.0e-5)
    assert_equal(mol0.bonds, mol1.bonds)


def test_load_dump_consistency(tmpdir):
    with as_file(files("iodata.test.data").joinpath("caffeine.mol2")) as fn_mol2:
        check_load_dump_consistency(tmpdir, fn_mol2)
    with as_file(files("iodata.test.data").joinpath("benzene.mol2")) as fn_mol2:
        check_load_dump_consistency(tmpdir, fn_mol2)


def test_load_many():
    with as_file(files("iodata.test.data").joinpath("caffeine.mol2")) as fn_mol2:
        mols = list(load_many(str(fn_mol2)))
    assert len(mols) == 2
    check_example(mols[0])
    assert mols[1].title == "ZINC00001085"
    assert mols[1].natom == 24
    assert_allclose(mols[0].atcoords[0] / angstrom, [-0.0178, 1.4608, 0.0101])
    assert_allclose(mols[1].atcoords[0] / angstrom, [-0.0100, 1.5608, 0.0201])


def test_load_dump_many_consistency(tmpdir):
    with as_file(files("iodata.test.data").joinpath("caffeine.mol2")) as fn_mol2:
        mols0 = list(load_many(str(fn_mol2)))
    # write mol2 file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, "test")
    dump_many(mols0, fn_tmp, fmt="mol2")
    mols1 = list(load_many(fn_tmp, fmt="mol2"))
    assert len(mols0) == len(mols1)
    for mol0, mol1 in zip(mols0, mols1):
        assert mol0.title == mol1.title
        assert_equal(mol0.atnums, mol1.atnums)
        assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.0e-5)
        assert_equal(mol0.bonds, mol1.bonds)


def test_load_dump_wrong_bond_num(tmpdir):
    with as_file(files("iodata.test.data").joinpath("silioh3.mol2")) as fn_mol:
        mol = load_one(str(fn_mol))
    mol.bonds[0][2] = -1
    fn_tmp = os.path.join(tmpdir, "test.mol2")
    dump_one(mol, fn_tmp)
    mol2 = load_one(fn_tmp)
    assert mol2.bonds[0][2] == bond2num["un"]


def test_load_water_bonds_warning():
    with (
        as_file(files("iodata.test.data").joinpath("water.mol2")) as fn_mol,
        pytest.warns(LoadWarning, match="Cannot interpret bond type"),
    ):
        mol = load_one(fn_mol)
    assert_equal(mol.bonds, [[0, 1, bond2num["un"]], [0, 2, bond2num["un"]]])

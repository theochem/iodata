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
"""Test iodata.formats.xyz module."""

import os

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one, load_many, dump_one, dump_many
from ..utils import angstrom
from ..formats.xyz import DEFAULT_ATOM_COLUMNS
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
    assert_equal(mol.atnums, [1, 8, 1])
    # check bond length
    print(np.linalg.norm(mol.atcoords[0] - mol.atcoords[2]) / angstrom)
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[1]) / angstrom, 0.960, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.atcoords[2] - mol.atcoords[1]) / angstrom, 0.960, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[2]) / angstrom, 1.568, atol=1.e-3)


FCC_ATOM_COLUMNS = DEFAULT_ATOM_COLUMNS + [
    # Storing the atomic numbers as zs in the extras attribute makes sense
    # for testing.
    ("extra", "zs", (), int, int, "{:2d}".format),
    # Note that in IOData, the energy gradient is stored, which contains the
    # negative forces.
    ("atgradient", None, (3,), float,
     (lambda word: -float(word)),
     (lambda value: "{:15.10f}".format(-value)))
]


def test_load_fcc_columns():
    with path('iodata.test.data', 'al_fcc.xyz') as fn_xyz:
        mol = load_one(str(fn_xyz), atom_columns=FCC_ATOM_COLUMNS)
    assert "zs" in mol.extra
    assert mol.extra["zs"].dtype == int
    assert_equal(mol.extra["zs"], [13] * mol.natom)
    assert mol.atgradient.shape == (mol.natom, 3)
    assert_allclose(mol.atgradient[0, 0], -0.285831)
    assert_allclose(mol.atgradient[2, 1], 0.268537)
    assert_allclose(mol.atgradient[-1, -1], -0.928032)


def check_load_dump_consistency(tmpdir, fn, atom_columns=None):
    """Check if dumping and loading an XYZ file results in the same data."""
    if atom_columns is None:
        atom_columns = DEFAULT_ATOM_COLUMNS
    if fn.endswith(".xyz"):
        mol0 = load_one(str(fn), atom_columns=atom_columns)
    else:
        mol0 = load_one(str(fn))
    # write xyz file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test.xyz')
    dump_one(mol0, fn_tmp, atom_columns=atom_columns)
    mol1 = load_one(fn_tmp, atom_columns=atom_columns)
    # check two xyz files
    assert mol0.title == mol1.title
    for attrname, keyname, _shapesuffix, _dtype, _loadword, _dumpword in atom_columns:
        value0 = getattr(mol0, attrname)
        value1 = getattr(mol1, attrname)
        if keyname is not None:
            value0 = value0[keyname]
            value1 = value1[keyname]
        assert_allclose(value0, value1, atol=1e-5)


def test_load_dump_consistency(tmpdir):
    with path('iodata.test.data', 'ch3_hf_sto3g.fchk') as fn_fchk:
        check_load_dump_consistency(tmpdir, str(fn_fchk))


def test_dump_xyz_water_element(tmpdir):
    with path('iodata.test.data', 'water_element.xyz') as fn_xyz:
        check_load_dump_consistency(tmpdir, str(fn_xyz))


def test_dump_xyz_water_number(tmpdir):
    with path('iodata.test.data', 'water_number.xyz') as fn_xyz:
        check_load_dump_consistency(tmpdir, str(fn_xyz))


def test_dump_xyz_fcc(tmpdir):
    with path('iodata.test.data', 'al_fcc.xyz') as fn_xyz:
        check_load_dump_consistency(tmpdir, str(fn_xyz), FCC_ATOM_COLUMNS)


def test_load_many():
    with path('iodata.test.data', 'water_trajectory.xyz') as fn_xyz:
        mols = list(load_many(str(fn_xyz)))
    assert len(mols) == 5
    for imol, mol in enumerate(mols):
        assert mol.title == "Frame {}".format(imol)
        assert_equal(mol.atnums, [8, 1, 1])
        assert mol.atcoords.shape == (3, 3)
    assert_allclose(mols[0].atcoords[2] / angstrom, [2.864329, 0.114369, 3.3635])
    assert_allclose(mols[2].atcoords[0] / angstrom, [-0.232964, -0.789588, -3.247615])
    assert_allclose(mols[-1].atcoords[1] / angstrom, [-2.123423, -3.355326, -3.353739])


def test_load_many_dataset_emptylines():
    with path('iodata.test.data', 'dataset_blanklines.xyz') as fn_xyz:
        mols = list(load_many(str(fn_xyz)))
    assert len(mols) == 3
    # Check 1st item
    assert mols[0].title == "H2O molecule"
    assert_equal(mols[0].atnums, [8, 1, 1])
    assert mols[0].atcoords.shape == (3, 3)
    assert_allclose(mols[0].atcoords[0] / angstrom, [3.340669, 0.264248, 2.537911])
    assert_allclose(mols[0].atcoords[1] / angstrom, [3.390790, -0.626971, 2.171608])
    assert_allclose(mols[0].atcoords[2] / angstrom, [2.864329, 0.114369, 3.363500])
    # Check 2nd item
    assert mols[1].title == "N atom"
    assert_equal(mols[1].atnums, [7])
    assert mols[1].atcoords.shape == (1, 3)
    assert_allclose(mols[1].atcoords, np.array([[0.0, 0.0, 0.0]]))
    # Check 3rd item
    assert mols[2].title == "CH4 molecule"
    assert_equal(mols[2].atnums, [6, 1, 1, 1, 1])
    assert mols[2].atcoords.shape == (5, 3)
    assert_allclose(mols[2].atcoords[1] / angstrom, [0.5518800034, 0.7964000023, 0.4945200020])
    assert_allclose(mols[2].atcoords[-1] / angstrom, [-0.4574600021, 0.3858400016, -0.9084400049])


def test_load_dump_many_consistency(tmpdir):
    with path('iodata.test.data', 'water_trajectory.xyz') as fn_xyz:
        mols0 = list(load_many(str(fn_xyz)))
    # write xyz file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test')
    dump_many(mols0, fn_tmp, fmt='xyz')
    mols1 = list(load_many(fn_tmp, fmt='xyz'))
    assert len(mols0) == len(mols1)
    for mol0, mol1 in zip(mols0, mols1):
        assert mol0.title == mol1.title
        assert_equal(mol0.atnums, mol1.atnums)
        assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-5)

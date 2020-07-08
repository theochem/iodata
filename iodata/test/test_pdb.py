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
"""Test iodata.formats.pdb module."""

import os

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one, load_many, dump_one, dump_many
from ..utils import angstrom
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_water():
    # test pdb of water
    with path('iodata.test.data', 'water.pdb') as fn_pdb:
        mol = load_one(str(fn_pdb))
    check_water(mol)


def check_water(mol):
    """Test some things on a water file."""
    assert_equal(mol.atnums, [1, 8, 1])
    # check bond length
    print(np.linalg.norm(mol.atcoords[0] - mol.atcoords[2]) / angstrom)
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[1]) / angstrom, 0.9599, atol=1.e-4)
    assert_allclose(np.linalg.norm(
        mol.atcoords[2] - mol.atcoords[1]) / angstrom, 0.9599, atol=1.e-4)
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[2]) / angstrom, 1.568, atol=1.e-3)


def check_load_dump_consistency(tmpdir, fn):
    """Check if dumping and loading an PDB file results in the same data."""
    mol0 = load_one(str(fn))
    # write pdb file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test.pdb')
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    # check two pdb files
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-5)
    if mol0.atffparams:
        assert_equal(mol0.atffparams.get('attypes'), mol1.atffparams.get('attypes'))
        assert_equal(mol0.atffparams.get('restypes'), mol1.atffparams.get('restypes'))
        assert_equal(mol0.atffparams.get('resnums'), mol1.atffparams.get('resnums'))
    if mol0.extra:
        assert_equal(mol0.extra.get('occupancy'), mol1.extra.get('occupancy'))
        assert_equal(mol0.extra.get('bfactor'), mol1.extra.get('bfactor'))


def test_load_dump_consistency(tmpdir):
    with path('iodata.test.data', 'water.pdb') as fn_pdb:
        check_load_dump_consistency(tmpdir, fn_pdb)


def check_load_dump_xyz_consistency(tmpdir, fn):
    """Check if dumping PDB from an xyz file results in the same data."""
    mol0 = load_one(str(fn))
    # write xyz file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test.pdb')
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    # check two molecule classes to be the same
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-2)
    # check if the general restype and attype are correct
    restypes = mol1.atffparams.get('restypes')
    attypes = mol1.atffparams.get('attypes')
    assert restypes[0] == "XXX"
    assert attypes[0] == "H1"
    # check if resnums are correct
    resnums = mol1.atffparams.get('resnums')
    assert_equal(resnums[0], -1)


def test_load_dump_xyz_consistency(tmpdir):
    with path('iodata.test.data', 'water.xyz') as fn_xyz:
        check_load_dump_xyz_consistency(tmpdir, fn_xyz)


def test_load_peptide():
    # test pdb of small peptide
    with path('iodata.test.data', '2luv.pdb') as fn_pdb:
        mol = load_one(str(fn_pdb))
    check_peptide(mol)


def check_peptide(mol):
    """Test some things on a peptide file."""
    assert mol.title.startswith("INTEGRIN")
    assert_equal(len(mol.atnums), 547)
    restypes = mol.atffparams.get('restypes')
    assert restypes[0] == "LYS"
    assert restypes[-1] == "LYS"
    attypes = mol.atffparams.get('attypes')
    assert attypes[0] == "N"
    assert attypes[-1] == "O"
    resnums = mol.atffparams.get('resnums')
    assert_equal(resnums[0], 1)
    assert_equal(resnums[-1], 35)
    occupancy = mol.extra.get('occupancy')
    assert_allclose(occupancy[0], 1.00)
    bfactor = mol.extra.get('bfactor')
    assert_allclose(bfactor[-1], 0.00)


def test_load_many():
    with path('iodata.test.data', 'water_trajectory.pdb') as fn_pdb:
        mols = list(load_many(str(fn_pdb)))
    assert len(mols) == 5
    for mol in mols:
        assert_equal(mol.atnums, [8, 1, 1])
        assert mol.atcoords.shape == (3, 3)
    assert_allclose(mols[0].atcoords[2] / angstrom, [2.864, 0.114, 3.364])
    assert_allclose(mols[2].atcoords[0] / angstrom, [-0.233, -0.790, -3.248])
    assert_allclose(mols[-1].atcoords[1] / angstrom, [-2.123, -3.355, -3.354])


def test_load_dump_many_consistency(tmpdir):
    with path('iodata.test.data', 'water_trajectory.pdb') as fn_pdb:
        mols0 = list(load_many(str(fn_pdb)))
    # write pdb file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test')
    dump_many(mols0, fn_tmp, fmt='pdb')
    mols1 = list(load_many(fn_tmp, fmt='pdb'))
    assert len(mols0) == len(mols1)
    for mol0, mol1 in zip(mols0, mols1):
        assert mol0.title == mol1.title
        assert_equal(mol0.atnums, mol1.atnums)
        assert_allclose(mol0.atcoords, mol1.atcoords, atol=1.e-5)

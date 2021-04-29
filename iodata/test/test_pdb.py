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
import pytest

from ..api import load_one, load_many, dump_one, dump_many
from ..utils import angstrom, FileFormatWarning
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


@pytest.mark.parametrize("case", ["single", "single_model"])
def test_load_water(case):
    # test pdb of water
    with path('iodata.test.data', f'water_{case}.pdb') as fn_pdb:
        mol = load_one(str(fn_pdb))
    check_water(mol)


def test_load_water_no_end():
    # test pdb of water
    with path('iodata.test.data', 'water_single_no_end.pdb') as fn_pdb:
        with pytest.warns(FileFormatWarning, match="The END is not found"):
            mol = load_one(str(fn_pdb))
    check_water(mol)


def check_water(mol):
    """Test some things on a water file."""
    assert mol.title == "water"
    assert_equal(mol.atnums, [1, 8, 1])
    # check bond length
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[1]) / angstrom, 0.9599, atol=1.e-4)
    assert_allclose(np.linalg.norm(
        mol.atcoords[2] - mol.atcoords[1]) / angstrom, 0.9599, atol=1.e-4)
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[2]) / angstrom, 1.568, atol=1.e-3)
    assert_equal(mol.bonds[:, :2], [[0, 1], [1, 2]])


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
    if mol0.atffparams is not None:
        assert_equal(mol0.atffparams.get('attypes'), mol1.atffparams.get('attypes'))
        assert_equal(mol0.atffparams.get('restypes'), mol1.atffparams.get('restypes'))
        assert_equal(mol0.atffparams.get('resnums'), mol1.atffparams.get('resnums'))
    if mol0.extra is not None:
        assert_equal(mol0.extra.get('occupancies'), mol1.extra.get('occupancies'))
        assert_equal(mol0.extra.get('bfactors'), mol1.extra.get('bfactors'))
        assert_equal(mol0.extra.get('chainids'), mol1.extra.get('chainids'))
    if mol0.bonds is None:
        assert mol1.bonds is None
    else:
        assert_equal(mol0.bonds, mol1.bonds)


@pytest.mark.parametrize("fn_base", [
    "water_single.pdb",
    "water_single_model.pdb",
    "ch5plus.pdb",
    "2luv.pdb",
    "2bcw.pdb",
])
def test_load_dump_consistency(fn_base, tmpdir):
    with path('iodata.test.data', fn_base) as fn_pdb:
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
    assert mol1.extra.get("chainids") is None
    # check if resnums are correct
    resnums = mol1.atffparams.get('resnums')
    assert_equal(resnums[0], -1)
    # There should be no bonds
    assert mol1.bonds is None


def test_load_dump_xyz_consistency(tmpdir):
    with path('iodata.test.data', 'water.xyz') as fn_xyz:
        check_load_dump_xyz_consistency(tmpdir, fn_xyz)


def test_load_peptide_2luv():
    # test pdb of small peptide
    with path('iodata.test.data', '2luv.pdb') as fn_pdb:
        mol = load_one(str(fn_pdb))
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
    assert_allclose(mol.extra.get('occupancies'), np.ones(mol.natom))
    assert_allclose(mol.extra.get('bfactors'), np.zeros(mol.natom))
    assert_equal(mol.extra.get('chainids'), ['A'] * mol.natom)


@pytest.mark.parametrize("case", ['trajectory', 'trajectory_no_model'])
def test_load_many(case):
    with path('iodata.test.data', f"water_{case}.pdb") as fn_pdb:
        mols = list(load_many(str(fn_pdb)))
    assert len(mols) == 5
    for mol in mols:
        assert_equal(mol.atnums, [8, 1, 1])
        assert mol.atcoords.shape == (3, 3)
        assert mol.extra.get('chainids') is None
        assert_allclose(mol.extra.get('occupancies'), np.ones(3))
        assert_allclose(mol.extra.get('bfactors'), np.zeros(3))
    assert_allclose(mols[0].atcoords[2] / angstrom, [2.864, 0.114, 3.364])
    assert_allclose(mols[2].atcoords[0] / angstrom, [-0.233, -0.790, -3.248])
    assert_allclose(mols[-1].atcoords[1] / angstrom, [-2.123, -3.355, -3.354])


@pytest.mark.parametrize("case", ['trajectory', 'trajectory_no_model'])
def test_load_dump_many_consistency(case, tmpdir):
    with path('iodata.test.data', f"water_{case}.pdb") as fn_pdb:
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


def test_load_2bcw():
    # test pdb with multiple chains
    with path("iodata.test.data", "2bcw.pdb") as fn_pdb:
        mol = load_one(fn_pdb)
    assert mol.title == """\
COORDINATES OF THE N-TERMINAL DOMAIN OF RIBOSOMAL PROTEIN L11,C-
TERMINAL DOMAIN OF RIBOSOMAL PROTEIN L7/L12 AND A PORTION OF THE G'
DOMAIN OF ELONGATION FACTOR G, AS FITTED INTO CRYO-EM MAP OF AN
ESCHERICHIA COLI 70S*EF-G*GDP*FUSIDIC ACID COMPLEX"""
    assert mol.extra["compound"] == """\
MOL_ID: 1;
MOLECULE: 50S RIBOSOMAL PROTEIN L11;
CHAIN: A;
FRAGMENT: N-TERMINAL DOMAIN;
MOL_ID: 2;
MOLECULE: 50S RIBOSOMAL PROTEIN L7/L12;
CHAIN: B;
FRAGMENT: C-TERMINAL DOMAIN;
SYNONYM: L8;
MOL_ID: 3;
MOLECULE: ELONGATION FACTOR G;
CHAIN: C;
FRAGMENT: A PORTION OF G' DOMAIN';
SYNONYM: EF-G"""
    assert mol.natom == 191
    assert (mol.atnums == 6).all()
    assert (mol.atffparams["attypes"] == ["CA"] * mol.natom).all()
    assert (mol.atffparams["restypes"][:3] == ['GLN', 'ILE', 'LYS']).all()
    assert (mol.atffparams["restypes"][-4:] == ['LYS', 'ILE', 'THR', 'PRO']).all()
    assert_allclose(mol.atcoords[0, 2] / angstrom, -86.956)
    assert_allclose(mol.atcoords[190, 0] / angstrom, -24.547)
    assert_allclose(mol.extra.get('occupancies'), np.ones(mol.natom))
    assert (mol.extra["chainids"] == ["A"] * 65 + ["B"] * 68 + ["C"] * 58).all()


def test_load_ch5plus_bonds():
    with path("iodata.test.data", "ch5plus.pdb") as fn_pdb:
        mol = load_one(fn_pdb)
    assert_equal(mol.bonds[:, :2], [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])

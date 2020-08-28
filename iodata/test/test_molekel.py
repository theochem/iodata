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
# pylint: disable=unsubscriptable-object,no-member
"""Test iodata.formats.molekel module."""

import os

from numpy.testing import assert_equal, assert_allclose

from .common import (check_orthonormal, compare_mols, compute_mulliken_charges,
                     load_one_warning)
from ..basis import convert_conventions
from ..api import load_one, dump_one
from ..overlap import compute_overlap
from ..utils import angstrom


def compare_mols_diff_formats(mol1, mol2):
    """Compare two IOData objects loaded from different formats."""
    assert_equal(mol1.atnums, mol2.atnums)
    assert_equal(mol1.atcorenums, mol2.atcorenums)
    # Atol set to 1e-15 to be able to treat values with really low exponents as 0
    assert_allclose(mol1.atcoords, mol2.atcoords, rtol=1e-07, atol=1e-15)

    # wfn
    assert mol1.obasis.primitive_normalization == mol2.obasis.primitive_normalization
    permutation, signs = convert_conventions(mol1.obasis, mol2.obasis.conventions)
    assert mol1.mo.kind == mol2.mo.kind
    assert_allclose(mol1.mo.occs, mol2.mo.occs)
    assert_allclose(mol1.mo.coeffs[permutation] * signs.reshape(-1, 1), mol2.mo.coeffs, atol=1e-8)
    assert_allclose(mol1.mo.energies, mol2.mo.energies)
    # compute and compare Mulliken charges
    charges1 = compute_mulliken_charges(mol1)
    charges2 = compute_mulliken_charges(mol2)
    assert_allclose(charges1, charges2, rtol=0.0, atol=1.0e-6)


def check_load_dump_consistency(fn: str, tmpdir: str, match: str = None):
    """Check if data is preserved after dumping and loading a Molekel file.

    Parameters
    ----------
    fn
        The Molekel filename to load
    tmpdir
        The temporary directory to dump and load the file.
    match
        When given, loading the file is expected to raise a warning whose
        message string contains match.

    """
    mol1 = load_one_warning(fn, match=match)
    fn_tmp = os.path.join(tmpdir, 'foo.bar')
    dump_one(mol1, fn_tmp, fmt='molekel')
    mol2 = load_one(fn_tmp, fmt='molekel')
    form = fn.split('.')
    if 'molden' in form:
        compare_mols_diff_formats(mol1, mol2)
    elif 'fchk' in form:
        compare_mols_diff_formats(mol1, mol2)
    else:
        compare_mols(mol1, mol2)


def test_load_dump_consistency_h2(tmpdir):
    check_load_dump_consistency('h2_sto3g.mkl', tmpdir, match="ORCA")


def test_load_dump_consistency_ethanol(tmpdir):
    check_load_dump_consistency('ethanol.mkl', tmpdir, match="ORCA")


def test_load_dump_consistency_li2(tmpdir):
    check_load_dump_consistency('li2.mkl', tmpdir, match="ORCA")


def test_load_molden_dump_molekel_li2(tmpdir):
    check_load_dump_consistency('li2.molden.input', tmpdir, match="ORCA")


def test_load_fchk_dump_molekel_li2(tmpdir):
    check_load_dump_consistency('li2_g09_nbasis_indep.fchk', tmpdir)


def test_load_mkl_ethanol():
    mol = load_one_warning("ethanol.mkl", match="ORCA")

    # Direct checks with mkl file
    assert_equal(mol.atnums.shape, (9,))
    assert_equal(mol.atnums[0], 1)
    assert_equal(mol.atnums[4], 6)
    assert_equal(mol.atcoords.shape, (9, 3))
    assert_allclose(mol.atcoords[2, 1] / angstrom, 2.239037, atol=1.e-5)
    assert_allclose(mol.atcoords[5, 2] / angstrom, 0.948420, atol=1.e-5)
    assert_equal(mol.atcharges['mulliken'].shape, (9,))
    q = [0.143316, -0.445861, 0.173045, 0.173021, 0.024542, 0.143066, 0.143080, -0.754230, 0.400021]
    assert_allclose(mol.atcharges['mulliken'], q)
    assert mol.obasis.nbasis == 39
    assert_allclose(mol.obasis.shells[0].exponents[0], 18.731137000)
    assert_allclose(mol.obasis.shells[4].exponents[0], 7.868272400)
    assert_allclose(mol.obasis.shells[7].exponents[1], 2.825393700)
    # No correspondence due to correction of the normalization of
    # the primivitves:
    # assert_allclose(mol.obasis.shells[2].coeffs[1, 0], 0.989450608)
    # assert_allclose(mol.obasis.shells[2].coeffs[3, 0], 2.079187061)
    # assert_allclose(mol.obasis.shells[-1].coeffs[-1, -1], 0.181380684)
    assert_equal([shell.icenter for shell in mol.obasis.shells[:5]], [0, 0, 1, 1, 1])
    assert_equal([shell.angmoms[0] for shell in mol.obasis.shells[:5]], [0, 0, 0, 0, 1])
    assert_equal([shell.nprim for shell in mol.obasis.shells[:5]], [3, 1, 6, 3, 3])
    assert_equal(mol.mo.coeffs.shape, (39, 39))
    assert_equal(mol.mo.energies.shape, (39,))
    assert_equal(mol.mo.occs.shape, (39,))
    assert_equal(mol.mo.occs[:13], 2.0)
    assert_equal(mol.mo.occs[13:], 0.0)
    assert_allclose(mol.mo.energies[4], -1.0206976)
    assert_allclose(mol.mo.energies[-1], 2.0748685)
    assert_allclose(mol.mo.coeffs[0, 0], 0.0000119)
    assert_allclose(mol.mo.coeffs[1, 0], -0.0003216)
    assert_allclose(mol.mo.coeffs[-1, -1], -0.1424743)


def test_load_mkl_li2():
    mol = load_one_warning("li2.mkl", match="ORCA")
    assert_equal(mol.atcharges['mulliken'].shape, (2,))
    assert_allclose(mol.atcharges['mulliken'], [0.5, 0.5])
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_load_mkl_h2():
    mol = load_one_warning("h2_sto3g.mkl", match="ORCA")
    assert_equal(mol.atcharges['mulliken'].shape, (2,))
    assert_allclose(mol.atcharges['mulliken'], [0, 0])
    # check mo normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

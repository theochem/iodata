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
"""Test iodata.formats.molden module."""

import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from .common import compute_mulliken_charges, compare_mols, check_orthonormal
from ..api import load_one, dump_one
from ..basis import convert_conventions
from ..formats.molden import _load_low
from ..overlap import compute_overlap, OVERLAP_CONVENTIONS
from ..utils import LineIterator, angstrom


try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_molden_li2_orca():
    with path('iodata.test.data', 'li2.molden.input') as fn_molden:
        mol = load_one(str(fn_molden))

    # Checkt title
    assert mol.title == 'Molden file created by orca_2mkl for BaseName=li2'

    # Check geometry
    assert_equal(mol.atnums, [3, 3])
    assert_allclose(mol.mo.occsa[:4], [1, 1, 1, 0])
    assert_allclose(mol.mo.occsb[:4], [1, 1, 0, 0])
    assert_equal(mol.mo.irreps, ['1a'] * mol.mo.norb)
    assert_equal(mol.mo.irrepsa, ['1a'] * mol.mo.norba)
    assert_equal(mol.mo.irrepsb, ['1a'] * mol.mo.norbb)
    assert_allclose(mol.atcoords[1], [5.2912331750, 0.0, 0.0])

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)

    # Check Mulliken charges
    charges = compute_mulliken_charges(mol)
    expected_charges = np.array([0.5, 0.5])
    assert_allclose(charges, expected_charges, atol=1.e-5)


def test_load_molden_h2o_orca():
    with path('iodata.test.data', 'h2o.molden.input') as fn_molden:
        mol = load_one(str(fn_molden))

    # Checkt title
    assert mol.title == 'Molden file created by orca_2mkl for BaseName=h2o'

    # Check geometry
    assert_equal(mol.atnums, [8, 1, 1])
    assert_allclose(mol.mo.occs[:6], [2, 2, 2, 2, 2, 0])
    assert_equal(mol.mo.irreps, ['1a'] * mol.mo.norb)
    assert_allclose(mol.atcoords[2], [0.0, -0.1808833432, 1.9123825806])

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp, 1e-5)

    # Check Mulliken charges
    charges = compute_mulliken_charges(mol)
    expected_charges = np.array([-0.816308, 0.408154, 0.408154])
    assert_allclose(charges, expected_charges, atol=1.e-5)


def test_load_molden_nh3_molden_pure():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    with path('iodata.test.data', 'nh3_molden_pure.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check geometry
    assert_equal(mol.atnums, [7, 1, 1, 1])
    assert_allclose(mol.atcoords[0] / angstrom, [-0.007455, 0.044763, 0.054913])
    assert_allclose(mol.atcoords[2] / angstrom, [-0.313244, -0.879581, 0.283126])
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_low_nh3_molden_cart():
    with path('iodata.test.data', 'nh3_molden_cart.molden') as fn_molden:
        lit = LineIterator(str(fn_molden))
        data = _load_low(lit)
    obasis = data['obasis']
    assert obasis.nbasis == 52
    assert len(obasis.shells) == 24
    for shell in obasis.shells:
        assert shell.kinds == ['c']
        assert shell.ncon == 1
    for ishell in [0, 1, 2, 3, 9, 10, 11, 14, 15, 16, 19, 20, 21]:
        shell = obasis.shells[ishell]
        assert shell.angmoms == [0]
    for ishell in [4, 5, 6, 12, 13, 17, 18, 22, 23]:
        shell = obasis.shells[ishell]
        assert shell.angmoms == [1]
    for ishell in [7, 8]:
        shell = obasis.shells[ishell]
        assert shell.angmoms == [2]
    for shell in obasis.shells[:9]:
        assert shell.icenter == 0
    for shell in obasis.shells[9:14]:
        assert shell.icenter == 1
    for shell in obasis.shells[14:19]:
        assert shell.icenter == 2
    for shell in obasis.shells[19:]:
        assert shell.icenter == 3

    shell0 = obasis.shells[0]
    assert shell0.nprim == 8
    assert shell0.exponents.shape == (8, )
    assert_allclose(shell0.exponents[4], 0.2856000000E+02)
    assert shell0.coeffs.shape == (8, 1)
    assert_allclose(shell0.coeffs[4, 0], 0.2785706633E+00)
    shell7 = obasis.shells[7]
    assert shell7.nprim == 1
    assert shell7.exponents.shape == (1, )
    assert_allclose(shell7.exponents, [0.8170000000E+00])
    assert_allclose(shell7.coeffs, [[1.0]])
    assert shell7.coeffs.shape == (1, 1)
    shell19 = obasis.shells[19]
    assert shell19.nprim == 3
    assert shell19.exponents.shape == (3, )
    assert_allclose(shell19.exponents, [
        0.1301000000E+02, 0.1962000000E+01, 0.4446000000E+00])
    assert_allclose(shell19.coeffs, [
        [0.3349872639E-01], [0.2348008012E+00], [0.8136829579E+00]])
    assert shell19.coeffs.shape == (3, 1)

    assert data['mo'].coeffs.shape == (52, 52)
    assert_allclose(data['mo'].coeffs[:2, 0], [1.002730, 0.005420])
    assert_allclose(data['mo'].coeffs[-2:, 1], [0.003310, -0.011620])
    assert_allclose(data['mo'].coeffs[-4:-2, -1], [-0.116400, 0.098220])

    permutation, signs = convert_conventions(obasis, OVERLAP_CONVENTIONS)
    assert_equal(permutation, [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 14, 18, 15, 19,
        22, 23, 20, 24, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    assert_equal(signs, [1] * 52)


def test_load_molden_nh3_molden_cart():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    with path('iodata.test.data', 'nh3_molden_cart.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.3138, -0.4300, -0.0667, 0.1829])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_nh3_orca():
    # The file tested here is created with ORCA. It should be read in
    # properly by altering normalization and sign conventions.
    with path('iodata.test.data', 'nh3_orca.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_nh3_psi4():
    # The file tested here is created with PSI4 (pre 1.0). It should be read in
    # properly by altering normalization conventions.
    with path('iodata.test.data', 'nh3_psi4_1.0.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_nh3_psi4_1():
    # The file tested here is created with PSI4 (version 1.0).
    # It should be read in properly by renormalizing the contractions.
    with path('iodata.test.data', 'nh3_psi4_1.0.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_he2_ghost_psi4_1():
    # The file tested here is created with PSI4 (version 1.0).
    with path('iodata.test.data', 'he2_ghost_psi4_1.0.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    np.testing.assert_equal(mol.atcorenums, np.array([0.0, 2.0]))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([-0.0041, 0.0041])
    assert_allclose(charges, molden_charges, atol=5e-4)


def test_load_molden_nh3_molpro2012():
    # The file tested here is created with MOLPRO2012.
    with path('iodata.test.data', 'nh3_molpro2012.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_neon_turbomole():
    # The file tested here is created with Turbomole 7.1.
    with path('iodata.test.data',
              'neon_turbomole_def2-qzvp.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    charges = compute_mulliken_charges(mol)
    assert abs(charges).max() < 1e-3


def test_load_molden_nh3_turbomole():
    # The file tested here is created with Turbomole 7.1
    with path('iodata.test.data', 'nh3_turbomole.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Turbomole output.
    # These are slightly different than in the other tests because we are using
    # Cartesian functions.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.03801, -0.27428, 0.01206, 0.22421])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_f():
    with path('iodata.test.data', 'F.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    assert_allclose(mol.mo.occsa[:6], [1, 1, 1, 1, 1, 0])
    assert_allclose(mol.mo.occsb[:6], [1, 1, 1, 1, 0, 0])
    assert_equal(mol.mo.irrepsa[:6], ['Ag', 'Ag', 'B3u', 'B2u', 'B1u', 'B3u'])
    assert_equal(mol.mo.irrepsb[:6], ['Ag', 'Ag', 'B3u', 'B2u', 'B1u', 'B3u'])


def check_load_dump_consistency(fn, tmpdir):
    """Check if data is preserved after dumping and loading a Molden file.

    Parameters
    ----------
    fn : str
        The Molden filename to load
    tmpdir : str
        The temporary directory to dump and load the file.

    """
    with path('iodata.test.data', fn) as file_name:
        mol1 = load_one(str(file_name))
    fn_tmp = os.path.join(tmpdir, 'foo.bar')
    dump_one(mol1, fn_tmp, fmt='molden')
    mol2 = load_one(fn_tmp, fmt='molden')
    compare_mols(mol1, mol2)


def test_load_dump_consistency_h2o(tmpdir):
    check_load_dump_consistency('h2o.molden.input', tmpdir)


def test_load_dump_consistency_li2(tmpdir):
    check_load_dump_consistency('li2.molden.input', tmpdir)


def test_load_dump_consistency_f(tmpdir):
    check_load_dump_consistency('F.molden', tmpdir)


def test_load_dump_consistency_nh3_molden_pure(tmpdir):
    check_load_dump_consistency('nh3_molden_pure.molden', tmpdir)


def test_load_dump_consistency_nh3_molden_cart(tmpdir):
    check_load_dump_consistency('nh3_molden_cart.molden', tmpdir)


def test_load_dump_consistency_he2_ghost_psi4_1(tmpdir):
    check_load_dump_consistency('he2_ghost_psi4_1.0.molden', tmpdir)

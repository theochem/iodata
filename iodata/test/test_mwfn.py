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
"""Test iodata.formats.mwfn module."""

from importlib.resources import as_file, files

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from ..api import load_one
from ..overlap import compute_overlap


def load_helper(fn):
    """Load a test file with iodata.iodata.load_one."""
    with as_file(files("iodata.test.data").joinpath(fn)) as absfn:
        return load_one(absfn)


def test_load_mwfn_ch3_rohf_g03():
    mol = load_helper("ch3_rohf_sto3g_g03_fchk_multiwfn3.7.mwfn")
    assert_equal(mol.mo.occs.shape[0], mol.mo.coeffs.shape[1])
    assert_equal(mol.mo.occs.min(), 0.0)
    assert_equal(mol.mo.occs.max(), 2.0)
    assert_equal(mol.extra["full_virial_ratio"], 2.00174844)
    assert_equal(mol.extra["nindbasis"], 8)
    assert_equal(np.sum([shell.nprim * shell.nbasis for shell in mol.obasis.shells]), 24)
    assert_equal(len(mol.obasis.shells), 6)
    assert_equal(np.sum([shell.nprim for shell in mol.obasis.shells]), 18)
    assert_equal(mol.charge, 0.0)
    assert_equal(mol.nelec, 9)
    assert_equal(mol.natom, 4)
    assert_equal(mol.energy, -3.90732095e01)
    assert_allclose([shell.angmoms[0] for shell in mol.obasis.shells], [0, 0, 1, 0, 0, 0])
    assert_allclose([shell.icenter for shell in mol.obasis.shells], [0, 0, 0, 1, 2, 3])
    assert_allclose([shell.nprim for shell in mol.obasis.shells], [3, 3, 3, 3, 3, 3])
    exponents1 = np.array([7.16168373e01, 1.30450963e01, 3.53051216e00])
    exponents2 = np.array([2.94124936e00, 6.83483096e-01, 2.22289916e-01])
    exponents3 = np.array([2.94124936e00, 6.83483096e-01, 2.22289916e-01])
    exponents4 = np.array([3.42525091e00, 6.23913730e-01, 1.68855404e-01])
    assert_allclose(mol.obasis.shells[0].exponents, exponents1)
    assert_allclose(mol.obasis.shells[1].exponents, exponents2)
    assert_allclose(mol.obasis.shells[2].exponents, exponents3)
    assert_allclose(mol.obasis.shells[3].exponents, exponents4)
    assert_allclose(mol.obasis.shells[4].exponents, exponents4)
    assert_allclose(mol.obasis.shells[5].exponents, exponents4)
    coeffs1 = np.array([[1.54328967e-01], [5.35328142e-01], [4.44634542e-01]])
    coeffs2 = np.array([[-9.99672292e-02], [3.99512826e-01], [7.00115469e-01]])
    coeffs3 = np.array([[1.55916275e-01], [6.07683719e-01], [3.91957393e-01]])
    coeffs4 = np.array([[1.54328967e-01], [5.35328142e-01], [4.44634542e-01]])
    assert_allclose(mol.obasis.shells[0].coeffs, coeffs1)
    assert_allclose(mol.obasis.shells[1].coeffs, coeffs2)
    assert_allclose(mol.obasis.shells[2].coeffs, coeffs3)
    assert_allclose(mol.obasis.shells[3].coeffs, coeffs4)
    assert_allclose(mol.obasis.shells[4].coeffs, coeffs4)
    assert_allclose(mol.obasis.shells[5].coeffs, coeffs4)
    # test first molecular orbital information
    coeff = np.array(
        [
            9.92532359e-01,
            3.42148679e-02,
            3.30477771e-06,
            -1.97321450e-03,
            0.00000000e00,
            -6.94439001e-03,
            -6.94439001e-03,
            -6.94539905e-03,
        ]
    )
    assert_equal(mol.mo.coeffs[:, 0], coeff)
    mo_energies = np.array(
        [
            -1.09902284e01,
            -8.36918686e-01,
            -5.24254982e-01,
            -5.23802785e-01,
            -1.26686819e-02,
            6.64707810e-01,
            7.68278159e-01,
            7.69362712e-01,
        ]
    )
    assert_allclose(mol.mo.energies, mo_energies)
    assert_equal(mol.mo.occs[0], 2.000000)
    assert_equal(mol.extra["mo_sym"][0], "?")
    # test that for the same molecule fchk and mwfn generate the same objects.
    olp = compute_overlap(mol.obasis, mol.atcoords)
    mol2 = load_helper("ch3_rohf_sto3g_g03.fchk")
    olp_fchk = compute_overlap(mol2.obasis, mol2.atcoords)
    assert_allclose(mol.atcoords, mol2.atcoords, atol=1e-7, rtol=1e-7)
    assert_allclose(mol2.obasis.shells[0].coeffs, coeffs1)
    # Mind the gap, I mean... the SP contraction
    assert_allclose(mol2.obasis.shells[1].coeffs[:, 0], np.squeeze(coeffs2.T))
    assert_allclose(mol2.obasis.shells[1].coeffs[:, 1], np.squeeze(coeffs3.T))
    assert_allclose(mol2.obasis.shells[3].coeffs, coeffs4)
    assert_allclose(mol2.obasis.shells[4].coeffs, coeffs4)
    assert_allclose(olp, olp_fchk, atol=1e-7, rtol=1e-7)


def test_load_mwfn_ch3_hf_g03():
    mol = load_helper("ch3_hf_sto3g_fchk_multiwfn3.7.mwfn")
    assert_equal(mol.mo.occs.shape[0], mol.mo.coeffs.shape[1])
    assert_equal(mol.extra["wfntype"], 1)
    # test first molecular orbital information
    coeff = np.array(
        [
            9.91912304e-01,
            3.68365244e-02,
            9.23239012e-04,
            9.05953703e-04,
            9.05953703e-04,
            -7.36810756e-03,
            -7.36810756e-03,
            -7.36919429e-03,
        ]
    )
    assert_equal(mol.mo.coeffs[:, 0], coeff)
    mo_energies = np.array(
        [
            -1.10094534e01,
            -9.07622407e-01,
            -5.37709620e-01,
            -5.37273275e-01,
            -3.63936540e-01,
            6.48361367e-01,
            7.58140704e-01,
            7.59223157e-01,
            -1.09780991e01,
            -8.01569083e-01,
            -5.19454722e-01,
            -5.18988806e-01,
            3.28562907e-01,
            7.04456296e-01,
            7.88139770e-01,
            7.89228899e-01,
        ]
    )
    assert_allclose(mol.mo.energies, mo_energies)
    assert_equal(mol.mo.occs[0], 1.000000)
    assert_equal(mol.extra["mo_sym"][0], "?")
    # test that for the same molecule fchk and mwfn generate the same objects.
    olp = compute_overlap(mol.obasis, mol.atcoords)
    mol2 = load_helper("ch3_hf_sto3g.fchk")
    olp_fchk = compute_overlap(mol2.obasis, mol2.atcoords)
    assert_allclose(mol.atcoords, mol2.atcoords, atol=1e-7, rtol=1e-7)
    assert_allclose(olp, olp_fchk, atol=1e-7, rtol=1e-7)


def test_nelec_charge():
    mol1 = load_helper("ch3_rohf_sto3g_g03_fchk_multiwfn3.7.mwfn")
    assert mol1.nelec == 9
    assert mol1.charge == 0
    mol2 = load_helper("he_spdfgh_virtual_fchk_multiwfn3.7.mwfn")
    assert mol2.nelec == 2
    assert mol2.charge == 0
    mol3 = load_helper("ch3_hf_sto3g_fchk_multiwfn3.7.mwfn")
    assert mol3.nelec == 9
    assert mol3.charge == 0


def test_load_mwfn_he_spdfgh_g03():
    mol = load_helper("he_spdfgh_virtual_fchk_multiwfn3.7.mwfn")
    assert_equal(mol.mo.occs.shape[0], mol.mo.coeffs.shape[1])
    assert_equal(mol.extra["wfntype"], 0)
    # test first molecular orbital information
    coeff = np.array(
        [
            8.17125208e-01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            1.58772965e-02,
            1.58772965e-02,
            1.58772965e-02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            7.73667846e-02,
            0.00000000e00,
            4.53013505e-02,
            0.00000000e00,
            7.73667846e-02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            4.53013505e-02,
            0.00000000e00,
            4.53013505e-02,
            0.00000000e00,
            0.00000000e00,
            7.73667846e-02,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    )

    assert_equal(mol.mo.coeffs[:, 0], coeff)
    mo_energies = np.array(
        [
            -3.83109139e-01,
            6.72890652e-02,
            6.72890652e-02,
            6.72890652e-02,
            3.33282755e-01,
            5.51389775e-01,
            5.51389775e-01,
            5.51389775e-01,
            5.51389775e-01,
            5.51389775e-01,
            8.85311032e-01,
            8.85311032e-01,
            8.85311032e-01,
            1.19945800e00,
            1.37176438e00,
            1.37176438e00,
            1.37176438e00,
            1.37176438e00,
            1.37176438e00,
            1.37176438e00,
            1.37176438e00,
            1.89666973e00,
            1.89666973e00,
            1.89666973e00,
        ]
    )
    assert_allclose(mol.mo.energies[:24], mo_energies)
    # energies were truncated at 24 entries, this checks the last energy entry
    assert mol.mo.energies[55] == 6.12473238e00
    assert_equal(mol.mo.occs[0], 2.000000)
    assert_equal(mol.extra["mo_sym"][0], "?")
    # this tests thhe last of the molecular orbital entries
    assert_equal(mol.mo.occs[55], 0.000000)
    assert_equal(mol.extra["mo_sym"][55], "?")
    # test that for the same molecule fchk and mwfn generate the same objects.
    olp = compute_overlap(mol.obasis, mol.atcoords)
    mol2 = load_helper("he_spdfgh_virtual.fchk")
    olp_fchk = compute_overlap(mol2.obasis, mol2.atcoords)
    assert_allclose(mol.atcoords, mol2.atcoords, atol=1e-7, rtol=1e-7)
    assert_allclose(olp, olp_fchk, atol=1e-7, rtol=1e-7)

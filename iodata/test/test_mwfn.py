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

import os
import re
from importlib.resources import as_file, files

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import load_one
from ..formats.mwfn import MATRIX_HEADER_PATTERN
from ..overlap import compute_overlap
from ..utils import LoadError


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
    assert_equal(np.sum([shell.nexp * shell.nbasis for shell in mol.obasis.shells]), 24)
    assert_equal(len(mol.obasis.shells), 6)
    assert_equal(np.sum([shell.nexp for shell in mol.obasis.shells]), 18)
    assert_equal(mol.charge, 0.0)
    assert_equal(mol.nelec, 9)
    assert_equal(mol.natom, 4)
    assert_equal(mol.energy, -3.90732095e01)
    assert_allclose([shell.angmoms[0] for shell in mol.obasis.shells], [0, 0, 1, 0, 0, 0])
    assert_allclose([shell.icenter for shell in mol.obasis.shells], [0, 0, 0, 1, 2, 3])
    assert_allclose([shell.nexp for shell in mol.obasis.shells], [3, 3, 3, 3, 3, 3])
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


def test_load_mwfn_fractional_occupations(tmp_path):
    # Occupations are written with 6 decimals, so three alpha electrons of 1/3
    # add up to 4.999999 instead of Naelec = 5.
    occs = ["1.000000"] * 4 + ["0.333333"] * 3 + ["0.000000"]
    lines = files("iodata.test.data").joinpath("ch3_hf_sto3g_fchk_multiwfn3.7.mwfn")
    lines = lines.read_text().splitlines(keepends=True)
    ialpha = 0
    for i, line in enumerate(lines):
        if line.startswith("Type= 1"):
            iocc = next(j for j in range(i, len(lines)) if lines[j].startswith("Occ="))
            lines[iocc] = f"Occ=  {occs[ialpha]}\n"
            ialpha += 1
    assert ialpha == len(occs)
    fn = tmp_path / "fractional.mwfn"
    fn.write_text("".join(lines))
    mol = load_one(fn)
    assert_allclose(mol.mo.occsa, [float(occ) for occ in occs])
    assert_allclose(mol.nelec, 9, atol=1e-5)


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


def test_load_mwfn_matrices_ch3_hf():
    mol = load_helper("ch3_hf_sto3g_fchk_multiwfn3.7.mwfn")
    olp = mol.one_ints["olp"]
    assert olp.shape == (8, 8)
    assert_allclose(olp, olp.T, atol=1e-12)
    # The overlap in the file agrees with the one from the basis set.
    assert_allclose(olp, compute_overlap(mol.obasis, mol.atcoords), atol=1e-7, rtol=1e-7)
    # The density matrices hold the right number of electrons (5 alpha and 4 beta).
    dm_alpha = 0.5 * (mol.one_rdms["scf_ao"] + mol.one_rdms["scf_spin_ao"])
    dm_beta = 0.5 * (mol.one_rdms["scf_ao"] - mol.one_rdms["scf_spin_ao"])
    assert_allclose(np.einsum("ij,ij", dm_alpha, olp), 5.0, atol=1e-6)
    assert_allclose(np.einsum("ij,ij", dm_beta, olp), 4.0, atol=1e-6)
    assert_allclose(np.einsum("ij,ij", mol.one_rdms["scf_ao"], olp), mol.nelec, atol=1e-6)
    # The first elements of the lower triangle in the file.
    assert_allclose(dm_alpha[0, 0], 1.03984043, atol=1e-8)
    assert_allclose(dm_alpha[1, 0], -1.35459706e-01, atol=1e-8)
    assert_allclose(dm_alpha[1, 1], 5.35184883e-01, atol=1e-8)
    assert_allclose(dm_alpha, dm_alpha.T, atol=1e-12)
    # They also follow from the orbitals.
    coeffs = mol.mo.coeffsa[:, mol.mo.occsa > 0]
    assert_allclose(coeffs @ coeffs.T, dm_alpha, atol=1e-5)


def test_load_mwfn_matrices_ch3_rohf():
    mol = load_helper("ch3_rohf_sto3g_g03_fchk_multiwfn3.7.mwfn")
    olp = mol.one_ints["olp"]
    assert_allclose(olp, compute_overlap(mol.obasis, mol.atcoords), atol=1e-7, rtol=1e-7)
    assert_allclose(np.einsum("ij,ij", mol.one_rdms["scf_ao"], olp), mol.nelec, atol=1e-6)
    spin = np.einsum("ij,ij", mol.one_rdms["scf_spin_ao"], olp)
    assert_allclose(spin, mol.spinpol, atol=1e-6)


def test_load_mwfn_matrices_he_spdfgh_total_density():
    mol = load_helper("he_spdfgh_virtual_fchk_multiwfn3.7.mwfn")
    olp = mol.one_ints["olp"]
    assert olp.shape == (56, 56)
    assert_allclose(olp, compute_overlap(mol.obasis, mol.atcoords), atol=1e-7, rtol=1e-7)
    # Only a total density matrix is in the file.
    assert "scf_spin_ao" not in mol.one_rdms
    assert_allclose(np.einsum("ij,ij", mol.one_rdms["scf_ao"], olp), 2.0, atol=1e-6)


@pytest.mark.parametrize(
    ("header", "name", "nrow", "ncol", "lower"),
    [
        ("$Overlap matrix, dim=    8    8  lower= 1", "Overlap matrix", 8, 8, "1"),
        ("$Total density matrix, dim=   56   56  lower= 0", "Total density matrix", 56, 56, "0"),
        ("$Overlap matrix, dim=    8    8  lower= 1\r\n", "Overlap matrix", 8, 8, "1"),
        # The fields of five characters can touch each other.
        ("$Overlap matrix, dim=1000010000  lower= 1", "Overlap matrix", 10000, 10000, "1"),
        ("$Alpha density matrix, dim=99999    1  lower= 0", "Alpha density matrix", 99999, 1, "0"),
        ("$Beta density matrix, dim=   10   10  lower= 2", "Beta density matrix", 10, 10, "2"),
    ],
)
def test_matrix_header_pattern(header, name, nrow, ncol, lower):
    match = MATRIX_HEADER_PATTERN.match(header)
    assert match is not None
    assert match["name"] == name
    assert (int(match["nrow"]), int(match["ncol"])) == (nrow, ncol)
    assert match["lower"] == lower


@pytest.mark.parametrize(
    "header",
    [
        "$Overlap matrix, dim=8 8 lower= 1",
        "$Overlap matrix, dim=    8  lower= 1",
        "$Overlap matrix, dim=    a    8  lower= 1",
        "$Overlap matrix, dim=    8    8  lower=",
        "$Overlap matrix, dim=    8    8",
        "Overlap matrix, dim=    8    8  lower= 1",
        "$Overlap matrix dim=    8    8  lower= 1",
    ],
)
def test_matrix_header_pattern_no_match(header):
    assert MATRIX_HEADER_PATTERN.match(header) is None


def load_modified(tmpdir, fn, modify):
    """Load a test file after a modification of its text."""
    text = files("iodata.test.data").joinpath(fn).read_text(encoding="utf-8")
    new_text = modify(text)
    assert new_text != text
    fn_tmp = os.path.join(tmpdir, "modified.mwfn")
    with open(fn_tmp, "w", encoding="utf-8") as f:
        f.write(new_text)
    return load_one(fn_tmp)


CH3 = "ch3_hf_sto3g_fchk_multiwfn3.7.mwfn"
OVERLAP_HEADER = "$Overlap matrix, dim=    8    8  lower= 1"


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (OVERLAP_HEADER, "$Overlap matrix, dim=    8    8  lower= 2", "lower= 2"),
        (OVERLAP_HEADER, "$Overlap matrix, dim=    8    8  lower= x", "lower= x"),
        (OVERLAP_HEADER, "$Overlap matrix, dim=    7    7  lower= 1", "basis functions"),
        (OVERLAP_HEADER, "$Overlap matrix, dim=    8    9  lower= 1", "basis functions"),
        (OVERLAP_HEADER, "$Overlap matrix, dim=8 8 lower= 1", "header"),
    ],
)
def test_load_mwfn_matrices_errors(tmpdir, old, new, match):
    with pytest.raises(LoadError, match=match):
        load_modified(tmpdir, CH3, lambda text: text.replace(old, new))


def test_load_mwfn_matrices_other_matrix_skipped(tmpdir):
    # Matrices that are not loaded are skipped.
    mol = load_helper(CH3)
    mol_other = load_modified(
        tmpdir,
        CH3,
        lambda text: text.replace(
            OVERLAP_HEADER,
            "$Hamiltonian matrix, dim=    8    8  lower= 1\nnot a number\n\n" + OVERLAP_HEADER,
        ),
    )
    assert_allclose(mol_other.one_ints["olp"], mol.one_ints["olp"], atol=1e-12)
    assert_allclose(mol_other.one_rdms["scf_ao"], mol.one_rdms["scf_ao"], atol=1e-12)


def test_load_mwfn_matrices_no_beta(tmpdir):
    # There are beta electrons, so the beta density matrix cannot be missing.
    with pytest.raises(LoadError, match="beta density matrix is missing"):
        load_modified(
            tmpdir, CH3, lambda text: re.sub(r"\$Beta density matrix.*?\n\n", "", text, flags=re.S)
        )


def test_load_mwfn_matrices_multiconfiguration(tmpdir):
    # Wfntype 0, 1 and 2 are single-determinant wavefunctions with an SCF density matrix.
    mol = load_helper("he_spdfgh_virtual_fchk_multiwfn3.7.mwfn")
    assert "scf_ao" in mol.one_rdms
    assert "post_scf_ao" not in mol.one_rdms
    # For Wfntype 3, the total density matrix is not an SCF density matrix.
    mol_post = load_modified(
        tmpdir,
        "he_spdfgh_virtual_fchk_multiwfn3.7.mwfn",
        lambda text: text.replace("Wfntype=   0", "Wfntype=   3"),
    )
    assert "scf_ao" not in mol_post.one_rdms
    assert_allclose(mol_post.one_rdms["post_scf_ao"], mol.one_rdms["scf_ao"], atol=1e-12)

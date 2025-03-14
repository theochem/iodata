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
"""Test iodata.formats.molden module."""

import os
from contextlib import ExitStack
from importlib.resources import as_file, files

import attrs
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import dump_one, load_one
from ..basis import MolecularBasis, Shell
from ..convert import HORTON2_CONVENTIONS, convert_conventions, convert_to_segmented
from ..formats.molden import _load_low
from ..formats.molden import dump_one as molden_dump_one
from ..iodata import IOData
from ..orbitals import MolecularOrbitals
from ..overlap import OVERLAP_CONVENTIONS, compute_overlap
from ..utils import (
    DumpError,
    LineIterator,
    LoadWarning,
    PrepareDumpError,
    PrepareDumpWarning,
    angstrom,
)
from .common import (
    check_orthonormal,
    compare_mols,
    compute_mulliken_charges,
    create_generalized_orbitals,
)


@pytest.mark.slow
def test_load_molden_li2_orca():
    with (
        as_file(files("iodata.test.data").joinpath("li2.molden.input")) as fn_molden,
        pytest.warns(LoadWarning) as record,
    ):
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]

    # Checkt title
    assert mol.title == "Molden file created by orca_2mkl for BaseName=li2"

    # Check geometry
    assert_equal(mol.atnums, [3, 3])
    assert_allclose(mol.mo.occsa[:4], [1, 1, 1, 0])
    assert_allclose(mol.mo.occsb[:4], [1, 1, 0, 0])
    assert_equal(mol.mo.irreps, ["1a"] * mol.mo.norb)
    assert_equal(mol.mo.irrepsa, ["1a"] * mol.mo.norba)
    assert_equal(mol.mo.irrepsb, ["1a"] * mol.mo.norbb)
    assert_allclose(mol.atcoords[1], [5.2912331750, 0.0, 0.0])

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)

    # Check Mulliken charges
    charges = compute_mulliken_charges(mol)
    expected_charges = np.array([0.5, 0.5])
    assert_allclose(charges, expected_charges, atol=1.0e-5)


def test_load_molden_li2_orca_huge_threshold():
    with as_file(files("iodata.test.data").joinpath("li2.molden.input")) as fn_molden:
        # The threshold is set very high, which skip a correction for ORCA.
        load_one(str(fn_molden), norm_threshold=1e4)


def test_load_molden_h2o_orca():
    with (
        as_file(files("iodata.test.data").joinpath("h2o.molden.input")) as fn_molden,
        pytest.warns(LoadWarning) as record,
    ):
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]

    # Checkt title
    assert mol.title == "Molden file created by orca_2mkl for BaseName=h2o"

    # Check geometry
    assert_equal(mol.atnums, [8, 1, 1])
    assert_allclose(mol.mo.occs[:6], [2, 2, 2, 2, 2, 0])
    assert_equal(mol.mo.irreps, ["1a"] * mol.mo.norb)
    assert_allclose(mol.atcoords[2], [0.0, -0.1808833432, 1.9123825806])

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges
    charges = compute_mulliken_charges(mol)
    expected_charges = np.array([-0.816308, 0.408154, 0.408154])
    assert_allclose(charges, expected_charges, atol=1.0e-5)


def test_load_molden_nh3_molden_pure():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    with as_file(files("iodata.test.data").joinpath("nh3_molden_pure.molden")) as fn_molden:
        mol = load_one(str(fn_molden))
    # Check geometry
    assert_equal(mol.atnums, [7, 1, 1, 1])
    assert_allclose(mol.atcoords[0] / angstrom, [-0.007455, 0.044763, 0.054913])
    assert_allclose(mol.atcoords[2] / angstrom, [-0.313244, -0.879581, 0.283126])

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp, atol=1e-4)  # low precision in file

    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.0e-3)


def test_load_molden_low_nh3_molden_cart():
    with (
        as_file(files("iodata.test.data").joinpath("nh3_molden_cart.molden")) as fn_molden,
        LineIterator(str(fn_molden)) as lit,
    ):
        data = _load_low(lit)
    obasis = data["obasis"]
    assert obasis.nbasis == 52
    assert len(obasis.shells) == 24
    for shell in obasis.shells:
        assert shell.kinds == ["c"]
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
    assert shell0.nexp == 8
    assert shell0.exponents.shape == (8,)
    assert_allclose(shell0.exponents[4], 0.2856000000e02)
    assert shell0.coeffs.shape == (8, 1)
    assert_allclose(shell0.coeffs[4, 0], 0.2785706633e00)
    shell7 = obasis.shells[7]
    assert shell7.nexp == 1
    assert shell7.exponents.shape == (1,)
    assert_allclose(shell7.exponents, [0.8170000000e00])
    assert_allclose(shell7.coeffs, [[1.0]])
    assert shell7.coeffs.shape == (1, 1)
    shell19 = obasis.shells[19]
    assert shell19.nexp == 3
    assert shell19.exponents.shape == (3,)
    assert_allclose(shell19.exponents, [0.1301000000e02, 0.1962000000e01, 0.4446000000e00])
    assert_allclose(shell19.coeffs, [[0.3349872639e-01], [0.2348008012e00], [0.8136829579e00]])
    assert shell19.coeffs.shape == (3, 1)

    assert data["mo"].coeffs.shape == (52, 52)
    assert_allclose(data["mo"].coeffs[:2, 0], [1.002730, 0.005420])
    assert_allclose(data["mo"].coeffs[-2:, 1], [0.003310, -0.011620])
    assert_allclose(data["mo"].coeffs[-4:-2, -1], [-0.116400, 0.098220])

    permutation, signs = convert_conventions(obasis, OVERLAP_CONVENTIONS)
    assert_equal(
        permutation,
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            16,
            17,
            14,
            18,
            15,
            19,
            22,
            23,
            20,
            24,
            21,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
        ],
    )
    assert_equal(signs, [1] * 52)

    # Check normalization
    olp = compute_overlap(obasis, data["atcoords"])
    check_orthonormal(data["mo"].coeffs, olp, atol=1e-4)  # low precision in file


def test_load_molden_nh3_molden_cart():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    with as_file(files("iodata.test.data").joinpath("nh3_molden_cart.molden")) as fn_molden:
        mol = load_one(str(fn_molden))

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp, atol=1e-4)  # low precision in file

    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.3138, -0.4300, -0.0667, 0.1829])
    assert_allclose(charges, molden_charges, atol=1.0e-3)


# The file tested here is created with CFOUR 2.1.
@pytest.mark.parametrize(
    ("path", "should_warn"),
    [
        ("h_sonly_sph_cfour.molden", False),
        ("h_ponly_sph_cfour.molden", False),
        ("h_donly_sph_cfour.molden", True),
        ("h_fonly_sph_cfour.molden", True),
        ("h_gonly_sph_cfour.molden", True),
        ("h_sonly_cart_cfour.molden", False),
        ("h_ponly_cart_cfour.molden", False),
        ("h_donly_cart_cfour.molden", True),
        ("h_fonly_cart_cfour.molden", True),
        ("h_gonly_cart_cfour.molden", True),
        ("h2o_ccpvdz_cfour.molden", True),
    ],
)
def test_load_molden_cfour(path, should_warn):
    with ExitStack() as stack:
        fn_molden = stack.enter_context(as_file(files("iodata.test.data").joinpath(path)))
        if should_warn:
            stack.enter_context(pytest.warns(LoadWarning))
        mol = load_one(str(fn_molden))
    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)


def test_load_molden_nh3_orca():
    # The file tested here is created with ORCA. It should be read in
    # properly by altering normalization and sign conventions.
    with (
        as_file(files("iodata.test.data").joinpath("nh3_orca.molden")) as fn_molden,
        pytest.warns(LoadWarning) as record,
    ):
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.0e-3)


def test_load_molden_nh3_psi4():
    # The file tested here is created with PSI4 (pre 1.0). It should be read in
    # properly by altering normalization conventions.
    with (
        as_file(files("iodata.test.data").joinpath("nh3_psi4.molden")) as fn_molden,
        pytest.warns(LoadWarning) as record,
    ):
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "PSI4 < 1.0" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.0e-3)


def test_load_molden_nh3_psi4_1():
    # The file tested here is created with PSI4 (version 1.0).
    # It should be read in properly by renormalizing the contractions.
    with (
        as_file(files("iodata.test.data").joinpath("nh3_psi4_1.0.molden")) as fn_molden,
        pytest.warns(LoadWarning) as record,
    ):
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "unnormalized" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.0e-3)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["zn", "mn", "cuh"])
def test_load_molden_high_am_psi4(case):
    # The file tested here is created with PSI4 1.3.2.
    # This is a special case because it contains higher angular momenta than
    # officially supported by the Molden format. Most virtual orbitals were removed.
    source = files("iodata.test.data").joinpath(f"psi4_{case}_cc_pvqz_pure.molden")
    with as_file(source) as fn_molden, pytest.warns(LoadWarning) as record:
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "unnormalized" in record[0].message.args[0]
    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    if mol.mo.kind == "restricted":
        check_orthonormal(mol.mo.coeffs, olp)
    elif mol.mo.kind == "unrestricted":
        check_orthonormal(mol.mo.coeffsa, olp)
        check_orthonormal(mol.mo.coeffsb, olp)
    else:
        raise NotImplementedError


@pytest.mark.slow
@pytest.mark.parametrize("case", ["zn", "cuh"])
def test_load_molden_high_am_orca(case):
    # The file tested here is created with ORCA.
    # This is a special case because it contains higher angular momenta than
    # officially supported by the Molden format. Most virtual orbitals were removed.
    source = files("iodata.test.data").joinpath(f"orca_{case}_cc_pvqz_pure.molden")
    with as_file(source) as fn_molden, pytest.warns(LoadWarning) as record:
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "ORCA" in record[0].message.args[0]
    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    assert mol.mo.kind == "restricted"
    check_orthonormal(mol.mo.coeffs, olp)


def test_load_molden_he2_ghost_psi4_1():
    # The file tested here is created with PSI4 (version 1.0).
    with as_file(files("iodata.test.data").joinpath("he2_ghost_psi4_1.0.molden")) as fn_molden:
        mol = load_one(str(fn_molden))
    np.testing.assert_equal(mol.atcorenums, np.array([0.0, 2.0]))

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([-0.0041, 0.0041])
    assert_allclose(charges, molden_charges, atol=5e-4)


def test_load_molden_h2o_6_31g_d_cart_psi4():
    # The file tested here is created with PSI4 1.3.2. It should be read in
    # properly after fixing for errors in AO normalization conventions.
    source = files("iodata.test.data").joinpath("h2o_psi4_1.3.2_6-31G_d_cart.molden")
    with as_file(source) as fn_molden, pytest.warns(LoadWarning) as record:
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "PSI4 <= 1.3.2" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from PSI4 output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([-0.86514, 0.43227, 0.43288])
    assert_allclose(charges, molden_charges, atol=1.0e-5)


@pytest.mark.slow
def test_load_molden_nh3_aug_cc_pvqz_cart_psi4():
    # The file tested here is created with PSI4 1.3.2. It should be read in
    # properly after fixing for errors in AO normalization conventions.
    source = files("iodata.test.data").joinpath("nh3_psi4_1.3.2_aug_cc_pvqz_cart.molden")
    with as_file(source) as fn_molden, pytest.warns(LoadWarning) as record:
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "PSI4 <= 1.3.2" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from PSI4 output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([-0.74507, 0.35743, 0.24197, 0.14567])
    assert_allclose(charges, molden_charges, atol=1.0e-5)


def test_load_be_cisd_321g_psi4():
    # CISD singlet calculation on Beryllium
    with as_file(
        files("iodata.test.data").joinpath("be_cisd_321g_psi4_singlet.molden")
    ) as fn_molden:
        mol = load_one(str(fn_molden))

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from PSI4 output.
    charges = compute_mulliken_charges(mol)
    assert_allclose(charges, [0.0], atol=1.0e-5)

    # Check things related to occupation numbers
    assert_allclose(mol.nelec, 4.0)
    assert mol.spinpol == 0.0
    assert_allclose(mol.mo.nelec, 4.0)
    assert mol.mo.spinpol == 0.0
    assert_allclose(
        mol.mo.occsa[-4:],
        [0.03164123603158929, 0.031641236031589526, 0.9041716124281225, 0.9999438368325004],
    )
    assert_allclose(mol.mo.occsa, mol.mo.occsb)
    assert mol.mo.occs_aminusb is None
    assert_allclose(mol.mo.occs, 2 * mol.mo.occsa)


def test_load_molden_nh3_molpro2012():
    # The file tested here is created with MOLPRO2012.
    with as_file(files("iodata.test.data").joinpath("nh3_molpro2012.molden")) as fn_molden:
        mol = load_one(str(fn_molden))

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.0e-3)


@pytest.mark.slow
def test_load_molden_neon_turbomole():
    # The file tested here is created with Turbomole 7.1.
    source = files("iodata.test.data").joinpath("neon_turbomole_def2-qzvp.molden")
    with as_file(source) as fn_molden, pytest.warns(LoadWarning) as record:
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "Turbomole" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    charges = compute_mulliken_charges(mol)
    assert abs(charges).max() < 1e-3


def test_load_molden_nh3_turbomole():
    # The file tested here is created with Turbomole 7.1
    with (
        as_file(files("iodata.test.data").joinpath("nh3_turbomole.molden")) as fn_molden,
        pytest.warns(LoadWarning) as record,
    ):
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "Turbomole" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffs, olp)

    # Check Mulliken charges.
    # Comparison with numbers from the Turbomole output.
    # These are slightly different than in the other tests because we are using
    # Cartesian functions.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.03801, -0.27428, 0.01206, 0.22421])
    assert_allclose(charges, molden_charges, atol=1.0e-3)


def test_load_molden_f():
    with (
        as_file(files("iodata.test.data").joinpath("F.molden")) as fn_molden,
        pytest.warns(LoadWarning) as record,
    ):
        mol = load_one(str(fn_molden))
    assert len(record) == 1
    assert "PSI4" in record[0].message.args[0]

    # Check normalization
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp)
    check_orthonormal(mol.mo.coeffsb, olp)

    assert_allclose(mol.mo.occsa[:6], [1, 1, 1, 1, 1, 0])
    assert_allclose(mol.mo.occsb[:6], [1, 1, 1, 1, 0, 0])
    assert_equal(mol.mo.irrepsa[:6], ["Ag", "Ag", "B3u", "B2u", "B1u", "B3u"])
    assert_equal(mol.mo.irrepsb[:6], ["Ag", "Ag", "B3u", "B2u", "B1u", "B3u"])


@pytest.mark.parametrize(
    ("fn", "match", "allow_changes"),
    [
        ("h2o.molden.input", "ORCA", False),
        pytest.param("li2.molden.input", "ORCA", False, marks=pytest.mark.slow),
        ("F.molden", "PSI4", False),
        ("nh3_molden_pure.molden", None, False),
        ("nh3_molden_cart.molden", None, False),
        ("he2_ghost_psi4_1.0.molden", None, False),
        pytest.param("psi4_cuh_cc_pvqz_pure.molden", "unnormalized", False, marks=pytest.mark.slow),
        ("hf_sto3g.fchk", None, True),
        ("h_sto3g.fchk", None, False),
        ("ch3_rohf_sto3g_g03.fchk", None, True),
    ],
)
def test_load_dump_consistency(tmpdir, fn, match, allow_changes):
    with ExitStack() as stack:
        file_name = stack.enter_context(as_file(files("iodata.test.data").joinpath(fn)))
        if match is not None:
            stack.enter_context(pytest.warns(LoadWarning, match=match))
        mol1 = load_one(file_name)
    fn_tmp = os.path.join(tmpdir, "foo.bar")
    if allow_changes:
        with pytest.warns(PrepareDumpWarning):
            dump_one(mol1, fn_tmp, fmt="molden", allow_changes=True)
    else:
        dump_one(mol1, fn_tmp, fmt="molden")
    mol2 = load_one(fn_tmp, fmt="molden")
    # Remove and or fix some things in mol1 to make it compatible with what
    # can be read from a Molden file:
    # - Change basis of mol1 to segmented.
    mol1.obasis = convert_to_segmented(mol1.obasis)
    # - Set default irreps in mol1, if not present.
    if mol1.mo.irreps is None:
        mol1.mo = attrs.evolve(mol1.mo, irreps=["1a"] * mol1.mo.norb)
    # - Remove the one_rdms from mol1.
    mol1.one_rdms = {}
    compare_mols(mol1, mol2)


def test_generalized_orbitals():
    # The Molden format does not support generalized MOs
    data = create_generalized_orbitals()
    with pytest.raises(PrepareDumpError):
        dump_one(data, "generalized.molden")


def test_mixed_pure_cartesian(tmpdir):
    rng = np.random.default_rng(42)
    data = IOData(
        atnums=[1, 1],
        atcoords=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        obasis=MolecularBasis(
            [
                Shell(0, [2], ["c"], [1.0], [[1.0]]),
                Shell(0, [2], ["p"], [1.0], [[1.0]]),
            ],
            HORTON2_CONVENTIONS,
            "L2",
        ),
        mo=MolecularOrbitals(
            "restricted",
            norba=2,
            norbb=2,
            occs=[1.0, 0.0],
            energies=[-1.0, -0.5],
            coeffs=rng.uniform(0, 1, (11, 2)),
        ),
    )
    assert data.obasis.nbasis == data.mo.nbasis
    with open(os.path.join(tmpdir, "foo.molden"), "w") as fh, pytest.raises(DumpError):
        molden_dump_one(fh, data)

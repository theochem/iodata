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
"""Test iodata.formats.wfn module."""

import os
from importlib.resources import as_file, files

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ..api import dump_one, load_one
from ..formats.wfn import load_wfn_low
from ..overlap import compute_overlap
from ..utils import LineIterator, PrepareDumpError, PrepareDumpWarning
from .common import (
    check_orthonormal,
    compare_mols,
    compute_mulliken_charges,
    create_generalized_orbitals,
)

# TODO: removed density, kin, nucnuc checks


def helper_load_wfn_low(fn_wfn):
    """Load a testing Gaussian log file with iodata.formats.wfn.load_wfn_low."""
    with as_file(files("iodata.test.data").joinpath(fn_wfn)) as fn, LineIterator(str(fn)) as lit:
        return load_wfn_low(lit)


def test_load_wfn_low_he_s():
    data = helper_load_wfn_low("he_s_orbital.wfn")
    # unpack data
    title, atnums, atcoords, centers, type_assignments = data[:5]
    exponents, mo_count, occ_num, mo_energy, coefficients, energy, virial, _ = data[5:]
    assert title == "He atom - decontracted 6-31G basis set"
    assert_equal(atnums.shape, (1,))
    assert_equal(atnums, [2])
    assert_equal(atcoords.shape, (1, 3))
    assert_equal(centers.shape, (4,))
    assert_equal(type_assignments.shape, (4,))
    assert_equal(exponents.shape, (4,))
    assert_equal(mo_count.shape, (1,))
    assert_equal(mo_count, [1])
    assert_equal(occ_num.shape, (1,))
    assert_equal(mo_energy.shape, (1,))
    assert_equal(coefficients.shape, (4, 1))
    assert_equal(type_assignments, [0, 0, 0, 0])
    assert_equal(centers, [0, 0, 0, 0])
    assert_equal(occ_num, [2.0])
    assert_allclose(atcoords, np.array([[0.00, 0.00, 0.00]]))
    assert_allclose(exponents, [0.3842163e02, 0.5778030e01, 0.1241774e01, 0.2979640e00])
    assert_allclose(mo_energy, [-0.914127])
    expected = np.array([0.26139500e00, 0.41084277e00, 0.39372947e00, 0.14762025e00])
    assert_allclose(coefficients, expected.reshape(4, 1))
    assert_allclose(energy, -2.855160426155, atol=1.0e-5)
    assert_allclose(virial, 1.99994256, atol=1.0e-6)


def test_load_wfn_low_h2o():
    data = helper_load_wfn_low("h2o_sto3g.wfn")
    # unpack data
    title, atnums, atcoords, centers, type_assignments = data[:5]
    exponents, mo_count, occ_num, mo_energy, coefficients, energy, virial, _ = data[5:]
    assert title == "H2O Optimization"
    assert_equal(atnums.shape, (3,))
    assert_equal(atcoords.shape, (3, 3))
    assert_equal(centers.shape, (21,))
    assert_equal(type_assignments.shape, (21,))
    assert_equal(exponents.shape, (21,))
    assert_equal(mo_count.shape, (5,))
    assert_equal(occ_num.shape, (5,))
    assert_equal(mo_energy.shape, (5,))
    assert_equal(coefficients.shape, (21, 5))
    assert_equal(atnums, np.array([8, 1, 1]))
    assert_equal(centers[:15], np.zeros(15, int))
    assert_equal(centers[15:], np.array([1, 1, 1, 2, 2, 2]))
    assert_equal(type_assignments[:6], np.zeros(6))
    assert_equal(type_assignments[6:15], np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]))
    assert_equal(type_assignments[15:], np.zeros(6))
    assert_equal(mo_count, [1, 2, 3, 4, 5])
    assert_equal(np.sum(occ_num), 10.0)
    assert_equal(occ_num, [2.0, 2.0, 2.0, 2.0, 2.0])
    assert_allclose(
        atcoords,
        np.array(
            [
                [-4.44734101, 3.39697999, 0.00000000],
                [-2.58401495, 3.55136194, 0.00000000],
                [-4.92380519, 5.20496220, 0.00000000],
            ]
        ),
    )
    assert_allclose(exponents[:3], [0.1307093e03, 0.2380887e02, 0.6443608e01])
    assert_allclose(exponents[5:8], [0.3803890e00, 0.5033151e01, 0.1169596e01])
    assert_allclose(exponents[13:16], [0.1169596e01, 0.3803890e00, 0.3425251e01])
    assert_allclose(exponents[-1], 0.1688554e00)
    assert_allclose(mo_energy, np.sort(mo_energy))
    assert_allclose(mo_energy[:3], [-20.251576, -1.257549, -0.593857])
    assert_allclose(mo_energy[3:], [-0.459729, -0.392617])
    expected = [0.42273517e01, -0.99395832e00, 0.19183487e-11, 0.44235381e00, -0.57941668e-14]
    assert_allclose(coefficients[0], expected)
    assert_allclose(coefficients[6, 2], 0.83831599e00)
    assert_allclose(coefficients[10, 3], 0.65034846e00)
    assert_allclose(coefficients[17, 1], 0.12988055e-01)
    assert_allclose(coefficients[-1, 0], -0.46610858e-03)
    assert_allclose(coefficients[-1, -1], -0.33277355e-15)
    assert_allclose(energy, -74.965901217080, atol=1.0e-6)
    assert_allclose(virial, 2.00600239, atol=1.0e-6)


def check_wfn(fn_wfn, nbasis, energy, charges_mulliken):
    """Check that MO are orthonormal & energy and charges match expected values."""
    # load file
    with as_file(files("iodata.test.data").joinpath(fn_wfn)) as file_wfn:
        mol = load_one(str(file_wfn))
    # check number of basis functions
    assert mol.obasis.nbasis == nbasis
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1.0e-5)
    if mol.mo.kind == "unrestricted":
        check_orthonormal(mol.mo.coeffsb, olp, 1.0e-5)
    # check energy & atomic charges
    if energy is not None:
        assert_allclose(mol.energy, energy, rtol=0.0, atol=1.0e-5)
    if charges_mulliken is not None:
        charges = compute_mulliken_charges(mol)
        assert_allclose(charges_mulliken, charges, rtol=0.0, atol=1.0e-5)
    return mol


def test_load_wfn_h2o_sto3g_decontracted():
    charges = np.array([-0.546656, 0.273328, 0.273328])
    check_wfn("h2o_sto3g_decontracted.wfn", 21, -75.162231674351, charges)


def test_load_wfn_h2_ccpvqz_virtual():
    mol = check_wfn("h2_ccpvqz.wfn", 74, -1.133504568400, np.array([0.0, 0.0]))
    expect = [82.64000, 12.41000, 2.824000, 0.7977000, 0.2581000]
    assert_allclose(
        [shell.exponents[0] for shell in mol.obasis.shells[:5]], expect, rtol=0.0, atol=1.0e-6
    )
    expect = [-0.596838, 0.144565, 0.209605, 0.460401, 0.460401]
    assert_allclose(mol.mo.energies[:5], expect, rtol=0.0, atol=1.0e-6)
    expect = [12.859067, 13.017471, 16.405834, 25.824716, 26.100443]
    assert_allclose(mol.mo.energies[-5:], expect, rtol=0.0, atol=1.0e-6)
    assert_equal(mol.mo.occs[:5], [2, 0, 0, 0, 0])
    assert_equal(mol.mo.occs.sum(), 2)


def test_load_wfn_h2o_sto3g():
    check_wfn("h2o_sto3g.wfn", 21, -74.96590121708, np.array([-0.330532, 0.165266, 0.165266]))


def test_load_wfn_li_sp_virtual():
    mol = check_wfn("li_sp_virtual.wfn", 8, -3.712905542719, np.array([0.0]))
    assert_equal(mol.mo.occs[:8], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert_equal(mol.mo.occs[8:], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expect = [-0.087492, -0.080310, 0.158784, 0.158784, 1.078773, 1.090891, 1.090891, 49.643670]
    assert_allclose(mol.mo.energies[:8], expect, rtol=0.0, atol=1.0e-6)
    expect = [-0.079905, 0.176681, 0.176681, 0.212494, 1.096631, 1.096631, 1.122821, 49.643827]
    assert_allclose(mol.mo.energies[8:], expect, rtol=0.0, atol=1.0e-6)
    assert_equal(mol.mo.coeffs.shape, (8, 16))


def test_load_wfn_li_sp():
    mol = check_wfn("li_sp_orbital.wfn", 8, -3.712905542719, None)
    assert mol.title == "Li atom - using s & p orbitals"
    assert_equal([mol.mo.norba, mol.mo.norbb], [2, 1])
    assert_allclose(mol.mo.energies, [-0.087492, -0.080310, -0.079905], rtol=0.0, atol=1.0e-6)


def test_load_wfn_o2():
    mol = check_wfn("o2_uhf.wfn", 72, -149.664140769678, np.array([0.0, 0.0]))
    assert_equal([mol.mo.norba, mol.mo.norbb], [9, 7])


def test_load_wfn_o2_virtual():
    mol = check_wfn("o2_uhf_virtual.wfn", 72, -149.664140769678, np.array([0.0, 0.0]))
    # check MO occupation
    assert_equal(mol.mo.occs.shape, (88,))
    assert_allclose(mol.mo.occsa, [1.0] * 9 + [0.0] * 35)
    assert_allclose(mol.mo.occsb, [1.0] * 7 + [0.0] * 37)
    # check MO energies
    assert_equal(mol.mo.energies.shape, (88,))
    mo_energies_a = mol.mo.energiesa
    assert_allclose(mo_energies_a[0], -20.752000, rtol=0, atol=1.0e-6)
    assert_allclose(mo_energies_a[10], 0.179578, rtol=0, atol=1.0e-6)
    assert_allclose(mo_energies_a[-1], 51.503193, rtol=0, atol=1.0e-6)
    mo_energies_b = mol.mo.energiesb
    assert_allclose(mo_energies_b[0], -20.697027, rtol=0, atol=1.0e-6)
    assert_allclose(mo_energies_b[15], 0.322590, rtol=0, atol=1.0e-6)
    assert_allclose(mo_energies_b[-1], 51.535258, rtol=0, atol=1.0e-6)
    # check MO coefficients
    assert_equal(mol.mo.coeffs.shape, (72, 88))


def test_load_wfn_lif_fci():
    mol = check_wfn("lif_fci.wfn", 44, -107.0575700853, np.array([-0.645282, 0.645282]))
    assert_equal(mol.mo.occs.shape, (18,))
    assert_allclose(mol.mo.occs.sum(), 12.0, rtol=0.0, atol=1.0e-6)
    assert_allclose(mol.mo.occs[0], 2.0, rtol=0.0, atol=1.0e-6)
    assert_allclose(mol.mo.occs[10], 0.00128021, rtol=0.0, atol=1.0e-6)
    assert_allclose(mol.mo.occs[-1], 0.00000054, rtol=0.0, atol=1.0e-6)
    assert_equal(mol.mo.energies.shape, (18,))
    assert_allclose(mol.mo.energies[0], -26.09321253, rtol=0.0, atol=1.0e-7)
    assert_allclose(mol.mo.energies[15], 1.70096290, rtol=0.0, atol=1.0e-7)
    assert_allclose(mol.mo.energies[-1], 2.17434072, rtol=0.0, atol=1.0e-7)
    assert_equal(mol.mo.coeffs.shape, (44, 18))


def test_load_wfn_lih_cation_fci():
    mol = check_wfn("lih_cation_fci.wfn", 26, -7.7214366383, np.array([0.913206, 0.086794]))
    assert_equal(mol.atnums, [3, 1])
    assert_equal(mol.mo.occs.shape, (11,))
    assert_allclose(mol.mo.occs.sum(), 3.0, rtol=0.0, atol=1.0e-6)
    # assert abs(mol.mo.occsa.sum() - 1.5) < 1.e-6


def test_load_one_lih_cation_cisd():
    with as_file(files("iodata.test.data").joinpath("lih_cation_cisd.wfn")) as file_wfn:
        mol = load_one(str(file_wfn))
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == "unrestricted"
    assert mol.mo.norba == 11
    assert mol.mo.norbb == 11
    assert mol.mo.norb == 22
    assert_equal(mol.mo.occsa, [1.0] * 2 + [0.0] * 9)
    assert_equal(mol.mo.occsb, [1.0] + [0.0] * 10)
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)


def test_load_one_lih_cation_uhf():
    with as_file(files("iodata.test.data").joinpath("lih_cation_uhf.wfn")) as file_wfn:
        mol = load_one(str(file_wfn))
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == "unrestricted"
    assert mol.mo.norba == 2
    assert mol.mo.norbb == 1
    assert mol.mo.norb == 3
    assert_equal(mol.mo.occsa, [1.0, 1.0])
    assert_equal(mol.mo.occsb, [1.0])
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)


def test_load_one_lih_cation_rohf():
    with as_file(files("iodata.test.data").joinpath("lih_cation_rohf.wfn")) as file_wfn:
        mol = load_one(str(file_wfn))
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == "restricted"
    assert mol.mo.norba == 2
    assert mol.mo.norbb == 2
    assert mol.mo.norb == 2
    assert_equal(mol.mo.occs, [2.0, 1.0])
    assert_equal(mol.mo.occsa, [1.0, 1.0])
    assert_equal(mol.mo.occsb, [1.0, 0.0])
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)


@pytest.mark.slow
def test_load_one_cah110_hf_sto3g_g09():
    with as_file(files("iodata.test.data").joinpath("cah110_hf_sto3g_g09.wfn")) as file_wfn:
        mol = load_one(str(file_wfn))
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == "unrestricted"
    assert mol.mo.norba == 123
    assert mol.mo.norbb == 123
    assert mol.mo.norb == 246
    occs = np.zeros(246)
    occs[0] = 1.0
    occsa, occsb = occs[:123], occs[123:]
    assert_equal(mol.mo.occs, occs)
    assert_equal(mol.mo.occsa, occsa)
    assert_equal(mol.mo.occsb, occsb)
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)


def check_load_dump_consistency(fn: str, tmpdir: str, atol: float = 1.0e-6, allow_changes=False):
    """Check if data is preserved after dumping and loading a WFN file.

    Parameters
    ----------
    fn
        The filename to load
    tmpdir
        The temporary directory to dump and load the file.
    atol
        The absolute tolerance on the numerical comparisons.

    """
    with as_file(files("iodata.test.data").joinpath(fn)) as file_name:
        mol1 = load_one(str(file_name))
    fn_tmp = os.path.join(tmpdir, "foo.wfn")
    dump_one(mol1, fn_tmp, allow_changes=allow_changes)
    mol2 = load_one(fn_tmp)
    # compare Mulliken charges
    charges1 = compute_mulliken_charges(mol1)
    charges2 = compute_mulliken_charges(mol2)
    assert_allclose(charges1, charges2, atol=atol)
    if fn.endswith(".wfn"):
        compare_mols(mol1, mol2, atol=atol)


@pytest.mark.parametrize(
    "path",
    [
        "lih_cation_cisd.wfn",
        "lih_cation_uhf.wfn",
        "lih_cation_rohf.wfn",
        "h2o_sto3g.wfn",
        "h2o_sto3g_decontracted.wfn",
        "lif_fci.wfn",
        pytest.param("cah110_hf_sto3g_g09.wfn", marks=pytest.mark.slow),
        "li_sp_orbital.wfn",
        "li_sp_virtual.wfn",
        "he_s_orbital.wfn",
        "he_s_virtual.wfn",
        "he_p_orbital.wfn",
        "he_d_orbital.wfn",
        "he_sp_orbital.wfn",
        "he_spd_orbital.wfn",
        "he_spdf_orbital.wfn",
        "he_spdfgh_orbital.wfn",
        "he_spdfgh_virtual.wfn",
        "h2_ccpvqz.wfn",
        "o2_uhf.wfn",
        "o2_uhf_virtual.wfn",
    ],
)
def test_load_dump_consistency(path, tmpdir):
    check_load_dump_consistency(path, tmpdir)


def test_load_dump_consistency_from_fchk_h2o(tmpdir):
    with pytest.warns(PrepareDumpWarning):
        check_load_dump_consistency("h2o_sto3g.fchk", tmpdir, allow_changes=True)


def test_load_dump_consistency_from_molden_nh3(tmpdir):
    check_load_dump_consistency("nh3_molden_cart.molden", tmpdir)


def test_dump_one_pure_functions(tmpdir):
    with pytest.raises(PrepareDumpError):
        check_load_dump_consistency("water_ccpvdz_pure_hf_g03.fchk", tmpdir)


def test_generalized_orbitals():
    # The WFN format does not support generalized MOs
    data = create_generalized_orbitals()
    with pytest.raises(PrepareDumpError):
        dump_one(data, "generalized.wfn")

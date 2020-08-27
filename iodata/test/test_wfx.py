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

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one, dump_one
from ..formats.wfx import load_data_wfx, parse_wfx
from ..overlap import compute_overlap
from ..utils import LineIterator

from .common import (check_orthonormal, truncated_file, compare_mols,
                     compute_mulliken_charges, load_one_warning)

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def helper_load_data_wfx(fn_wfx):
    """Load a testing WFX file with iodata.formats.wfx.load_data_wfx."""
    with path('iodata.test.data', fn_wfx) as fx:
        lit = LineIterator(str(fx))
        return load_data_wfx(lit)


def check_load_dump_consistency(fn, tmpdir):
    """Check if data is preserved after dumping and loading a Wfx file.

    Parameters
    ----------
    fn : str
        The Molekel filename to load
    tmpdir : str
        The temporary directory to dump and load the file.
    """
    with path('iodata.test.data', fn) as file_name:
        mol1 = load_one(str(file_name), fmt='wfx')
    fn_tmp = os.path.join(tmpdir, 'foo.bar')
    dump_one(mol1, fn_tmp, fmt='wfx')
    mol2 = load_one(fn_tmp, fmt='wfx')
    compare_mols(mol1, mol2)
    # compare Mulliken charges
    charges1 = compute_mulliken_charges(mol1)
    charges2 = compute_mulliken_charges(mol2)
    assert_allclose(charges2, charges1, rtol=1.0e-7, atol=0.0)


def test_load_dump_consistency_water(tmpdir):
    check_load_dump_consistency('water_sto3g_hf.wfx', tmpdir)


def test_load_dump_consistency_h2(tmpdir):
    check_load_dump_consistency('h2_ub3lyp_ccpvtz.wfx', tmpdir)


def test_load_dump_consistency_lih_cation_cisd(tmpdir):
    check_load_dump_consistency('lih_cation_cisd.wfx', tmpdir)


def test_load_dump_consistency_lih_cation_uhf(tmpdir):
    check_load_dump_consistency('lih_cation_uhf.wfx', tmpdir)


def test_load_dump_consistency_lih_cation_rohf(tmpdir):
    check_load_dump_consistency('lih_cation_rohf.wfx', tmpdir)


def compare_mulliken_charges(
        fname: str, tmpdir: str, rtol: float = 1.0e-7, atol: float = 0.0,
        match: str = None):
    """Check if charges are computed correctly after dumping and loading WFX file format.

    Parameters
    ----------
    fname
        The filename to be load.
    tmpdir
        The temporary directory to dump and load the file.
    rtol
        Relative tolerance when comparing charges. (optional)
    atol
        Absolute tolerance when comparing charges. (optional)
    match
        When given, loading the file is expected to raise a warning whose
        message string contains match.

    """
    mol1 = load_one_warning(fname, match=match)
    # dump WFX and check that file exists
    fn_tmp = os.path.join(tmpdir, f"{fname}.wfx")
    dump_one(mol1, fn_tmp)
    assert os.path.isfile(fn_tmp)
    # load dumped file and compare Mulliken charges
    mol2 = load_one(fn_tmp)
    charges1 = compute_mulliken_charges(mol1)
    charges2 = compute_mulliken_charges(mol2)
    assert_allclose(charges1, charges2, rtol=rtol, atol=atol)


def test_dump_one_from_fchk_h2o(tmpdir):
    compare_mulliken_charges('h2o_sto3g.fchk', tmpdir)
    compare_mulliken_charges('water_hfs_321g.fchk', tmpdir)
    compare_mulliken_charges('water_sto3g_hf_g03.fchk', tmpdir)


def test_dump_one_from_fchk_ch3_restricted(tmpdir):
    compare_mulliken_charges('ch3_hf_sto3g.fchk', tmpdir)


def test_dump_one_from_fchk_ch3_unrestricted(tmpdir):
    compare_mulliken_charges('ch3_rohf_sto3g_g03.fchk', tmpdir)


def test_dump_one_from_wfn_h2o(tmpdir):
    compare_mulliken_charges('h2o_sto3g.wfn', tmpdir)
    compare_mulliken_charges('h2o_sto3g_decontracted.wfn', tmpdir)


def test_dump_one_from_wfn_o2(tmpdir):
    compare_mulliken_charges('o2_uhf.wfn', tmpdir)
    compare_mulliken_charges('o2_uhf_virtual.wfn', tmpdir)


def test_dump_one_from_wfn_atom(tmpdir):
    # Li atom
    compare_mulliken_charges('li_sp_orbital.wfn', tmpdir)
    compare_mulliken_charges('li_sp_virtual.wfn', tmpdir)
    # He atom
    compare_mulliken_charges('he_s_orbital.wfn', tmpdir)
    compare_mulliken_charges('he_s_virtual.wfn', tmpdir)
    compare_mulliken_charges('he_p_orbital.wfn', tmpdir)
    compare_mulliken_charges('he_d_orbital.wfn', tmpdir)
    compare_mulliken_charges('he_sp_orbital.wfn', tmpdir)
    compare_mulliken_charges('he_spd_orbital.wfn', tmpdir)
    compare_mulliken_charges('he_spdf_orbital.wfn', tmpdir)
    compare_mulliken_charges('he_spdfgh_orbital.wfn', tmpdir)
    compare_mulliken_charges('he_spdfgh_virtual.wfn', tmpdir)


def test_dump_one_from_wfn_lih(tmpdir):
    compare_mulliken_charges('lih_cation_uhf.wfn', tmpdir)
    compare_mulliken_charges('lih_cation_rohf.wfn', tmpdir)
    compare_mulliken_charges('lih_cation_cisd.wfn', tmpdir)
    compare_mulliken_charges('lih_cation_fci.wfn', tmpdir)


def test_dump_one_from_wfn_lif(tmpdir):
    compare_mulliken_charges('lif_fci.wfn', tmpdir)


def test_dump_one_from_molden_h2o(tmpdir):
    compare_mulliken_charges('h2o.molden.input', tmpdir, match="ORCA")


def test_dump_one_from_molden_he2(tmpdir):
    compare_mulliken_charges('he2_ghost_psi4_1.0.molden', tmpdir)


def test_dump_one_from_molden_neon(tmpdir):
    compare_mulliken_charges(
        'neon_turbomole_def2-qzvp.molden', tmpdir, atol=1.0e-10, match="Turbomole")


def test_dump_one_from_molden_nh3(tmpdir):
    compare_mulliken_charges('nh3_molden_cart.molden', tmpdir)
    compare_mulliken_charges('nh3_molpro2012.molden', tmpdir)
    compare_mulliken_charges('nh3_turbomole.molden', tmpdir, match="Turbomole")


def test_dump_one_from_mkl_methanol(tmpdir):
    compare_mulliken_charges('ethanol.mkl', tmpdir, match="ORCA")


def test_dump_one_from_mkl_h2(tmpdir):
    compare_mulliken_charges('h2_sto3g.mkl', tmpdir, match="ORCA")


# add this test when pure to Cartesian basis set conversion is supported
# def test_dump_one_from_mkl_li2(tmpdir):
#     compare_mulliken_charges('li2.mkl', tmpdir)


def test_load_data_wfx_h2():
    """Test load_data_wfx with h2_ub3lyp_ccpvtz.wfx."""
    data = helper_load_data_wfx('h2_ub3lyp_ccpvtz.wfx')
    # check loaded data
    assert data['title'] == 'h2 ub3lyp/cc-pvtz opt-stable-freq'
    assert data['keywords'] == 'GTO'
    # assert model_name is None
    assert data['num_atoms'] == 2
    assert data['num_primitives'] == 34
    assert data['num_occ_mo'] == 56
    assert data['num_perturbations'] == 0
    assert data['num_electrons'] == 2
    assert data['num_alpha_electron'] == 1
    assert data['num_beta_electron'] == 1
    assert data['charge'] == 0.0
    assert data['spin_multi'] == 1
    assert_allclose(data['energy'], -1.179998789924e+00)
    assert_allclose(data['virial_ratio'], 2.036441983763e+00)
    assert_allclose(data['nuc_viral'], 1.008787649881e-08)
    assert_allclose(data['full_virial_ratio'], 2.036441992623e+00)
    assert_equal(data['nuclear_names'], ['H1', 'H2'])
    assert_equal(data['atnums'], np.array([1, 1]))
    assert_equal(data['mo_spins'], np.array(['Alpha'] * 28 + ['Beta'] * 28).T)
    coords = np.array([[0., 0., 0.7019452462164], [0., 0., -0.7019452462164]])
    assert_allclose(data['atcoords'], coords)
    assert_allclose(data['centers'], np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    assert_allclose(data['types'], np.array([
        1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    assert_allclose(data['exponents'], np.array([
        3.387000000000e+01, 5.095000000000e+00, 1.159000000000e+00,
        3.258000000000e-01, 1.027000000000e-01, 1.407000000000e+00,
        1.407000000000e+00, 1.407000000000e+00, 3.880000000000e-01,
        3.880000000000e-01, 3.880000000000e-01, 1.057000000000e+00,
        1.057000000000e+00, 1.057000000000e+00, 1.057000000000e+00,
        1.057000000000e+00, 1.057000000000e+00, 3.387000000000e+01,
        5.095000000000e+00, 1.159000000000e+00, 3.258000000000e-01,
        1.027000000000e-01, 1.407000000000e+00, 1.407000000000e+00,
        1.407000000000e+00, 3.880000000000e-01, 3.880000000000e-01,
        3.880000000000e-01, 1.057000000000e+00, 1.057000000000e+00,
        1.057000000000e+00, 1.057000000000e+00, 1.057000000000e+00,
        1.057000000000e+00]))
    assert_allclose(data['mo_occs'], ([1] + [0] * 27) * 2)
    assert_allclose(data['mo_energies'], np.array([
        -4.340830854172e-01, 5.810590098068e-02, 1.957476339319e-01,
        4.705943952631e-01, 5.116003517961e-01, 5.116003517961e-01,
        9.109680450208e-01, 9.372078887497e-01, 9.372078887497e-01,
        1.367198523024e+00, 2.035656924620e+00, 2.093459617091e+00,
        2.882582109554e+00, 2.882582109559e+00, 3.079758295551e+00,
        3.079758295551e+00, 3.356387932344e+00, 3.600856684661e+00,
        3.600856684661e+00, 3.793185027287e+00, 3.793185027400e+00,
        3.807665977092e+00, 3.807665977092e+00, 4.345665616275e+00,
        5.386560784523e+00, 5.386560784523e+00, 5.448122593462e+00,
        6.522366660004e+00, -4.340830854172e-01, 5.810590098068e-02,
        1.957476339319e-01, 4.705943952631e-01, 5.116003517961e-01,
        5.116003517961e-01, 9.109680450208e-01, 9.372078887497e-01,
        9.372078887497e-01, 1.367198523024e+00, 2.035656924620e+00,
        2.093459617091e+00, 2.882582109554e+00, 2.882582109559e+00,
        3.079758295551e+00, 3.079758295551e+00, 3.356387932344e+00,
        3.600856684661e+00, 3.600856684661e+00, 3.793185027287e+00,
        3.793185027400e+00, 3.807665977092e+00, 3.807665977092e+00,
        4.345665616275e+00, 5.386560784523e+00, 5.386560784523e+00,
        5.448122593462e+00, 6.522366660004e+00]))
    assert_allclose(data['atgradient'], [
        [9.744384163503e-17, -2.088844408785e-16, -7.185657679987e-09],
        [-9.744384163503e-17, 2.088844408785e-16, 7.185657679987e-09]])
    assert data['mo_coeffs'].shape == (34, 56)
    assert_allclose(data['mo_coeffs'][:, 0], [
        5.054717669172e-02, 9.116391481072e-02, 1.344211391235e-01,
        8.321037376208e-02, 1.854203733451e-02, -5.552096650015e-17,
        1.685043781907e-17, -2.493514848195e-02, 5.367769875676e-18,
        -8.640401342563e-21, -4.805966923740e-03, -3.124765025063e-04,
        -3.124765025063e-04, 6.249530050126e-04, 6.560467295881e-16,
        -8.389003686496e-17, 1.457172009403e-16, 5.054717669172e-02,
        9.116391481072e-02, 1.344211391235e-01, 8.321037376215e-02,
        1.854203733451e-02, 1.377812848830e-16, -5.365229184139e-18,
        2.493514848197e-02, -2.522774106094e-17, 2.213188439119e-17,
        4.805966923784e-03, -3.124765025186e-04, -3.124765025186e-04,
        6.249530050373e-04, -6.548275062740e-16, 4.865003740982e-17,
        -1.099855647247e-16])
    assert_allclose(data['mo_coeffs'][2, 9], 1.779549601504e-02)
    assert_allclose(data['mo_coeffs'][19, 14], -1.027984391469e-15)
    assert_allclose(data['mo_coeffs'][26, 36], -5.700424557682e-01)


def test_load_data_wfx_water():
    """Test load_data_wfx with water_sto3g_hf.wfx."""
    data = helper_load_data_wfx('water_sto3g_hf.wfx')
    # check loaded data
    assert data['title'] == 'H2O HF/STO-3G//HF/STO-3G'
    assert data['keywords'] == 'GTO'
    assert data['model_name'] == 'Restricted HF'
    assert data['num_atoms'] == 3
    assert data['num_primitives'] == 21
    assert data['num_occ_mo'] == 5
    assert data['num_perturbations'] == 0
    assert data['num_electrons'] == 10
    assert data['num_alpha_electron'] == 5
    assert data['num_beta_electron'] == 5
    assert data['charge'] == 0.0
    # assert_equal(num_spin_multi, np.array(None))
    assert_allclose(data['energy'], -7.49659011707870E+001)
    assert_allclose(data['virial_ratio'], 2.00599838291596E+000)
    # assert_allclose(data['nuclear_virial'], np.array(None))
    assert_allclose(data['full_virial_ratio'], 2.00600662884992E+000)
    assert_equal(data['nuclear_names'], ['O1', 'H2', 'H3'])
    assert_equal(data['atnums'], np.array([8, 1, 1]))
    assert_equal(data['mo_spins'], np.array(['Alpha and Beta'] * 5).T)
    assert_allclose(data['atcoords'], [
        [0.00000000000000, 0.00000000000000, 2.40242907000000E-1],
        [0.00000000000000, 1.43244242000000, -9.60971627000000E-1],
        [-1.75417809000000e-16, -1.43244242000000, -9.60971627000000E-1]])
    assert_allclose(data['centers'], np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
    assert_allclose(data['types'], np.array(
        [1, 1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 1, 1, 1, 1, 1, 1]))
    assert_allclose(data['exponents'], np.array([
        1.30709321000000E+002, 2.38088661000000E+001, 6.44360831000000E+000,
        5.03315132000000E+000, 1.16959612000000E+000, 3.80388960000000E-001,
        5.03315132000000E+000, 5.03315132000000E+000, 5.03315132000000E+000,
        1.16959612000000E+000, 1.16959612000000E+000, 1.16959612000000E+000,
        3.80388960000000E-001, 3.80388960000000E-001, 3.80388960000000E-001,
        3.42525091000000E+000, 6.23913730000000E-001, 1.68855404000000E-001,
        3.42525091000000E+000, 6.23913730000000E-001, 1.68855404000000E-001]))
    assert_allclose(data['mo_occs'], np.array([
        2.00000000000000E+000, 2.00000000000000E+000,
        2.00000000000000E+000, 2.00000000000000E+000,
        2.00000000000000E+000]))
    assert_allclose(data['mo_energies'], np.array([
        -2.02515479000000E+001, -1.25760928000000E+000, -5.93941119000000E-001,
        -4.59728723000000E-001, -3.92618460000000E-001]))
    assert_allclose(data['atgradient'][:2, :], [
        [6.09070231000000E-016, -5.55187875000000E-016, -2.29270172000000E-004],
        [-2.46849911000000E-016, -1.18355659000000E-004, 1.14635086000000E-004]])
    assert data['mo_coeffs'].shape == (21, 5)
    assert_allclose(data['mo_coeffs'][:, 0], [
        4.22735025664585E+000, 4.08850914632625E+000, 1.27420971692421E+000,
        -6.18883321546465E-003, 8.27806436882009E-003, 6.24757868903820E-003,
        0.00000000000000E+000, 0.00000000000000E+000, -6.97905144921135E-003,
        0.00000000000000E+000, 0.00000000000000E+000, -4.38861481239680E-003,
        0.00000000000000E+000, 0.00000000000000E+000, -6.95230322147800E-004,
        -1.54680714141406E-003, -1.49600452906993E-003, -4.66239267760156E-004,
        -1.54680714141406E-003, -1.49600452906993E-003, -4.66239267760156E-004
    ], rtol=0.0, atol=1.e-8)
    assert_allclose(data['mo_coeffs'][1, 3], -4.27845789719456E-001)


def test_parse_wfx_missing_tag_h2o():
    """Check that missing sections result in an exception."""
    with path('iodata.test.data', 'water_sto3g_hf.wfx') as fn_wfx:
        lit = LineIterator(fn_wfx)
        with pytest.raises(IOError) as error:
            parse_wfx(lit, required_tags=["<Foo Bar>"])
    assert str(error.value).endswith("Section <Foo Bar> is missing from loaded WFX data.")


def test_load_data_wfx_h2o_error():
    """Check that sections without a closing tag result in an exception."""
    with path('iodata.test.data', 'h2o_error.wfx') as fn_wfx:
        with pytest.raises(IOError) as error:
            load_one(str(fn_wfx))
    assert str(error.value).endswith(
        "Expecting line </Number of Nuclei> but got </Number of Primitives>.")


def test_load_truncated_h2o(tmpdir):
    """Check that a truncated file raises an exception."""
    with path('iodata.test.data', 'water_sto3g_hf.wfx') as fn_wfx:
        with truncated_file(str(fn_wfx), 152, 0, tmpdir) as fn_truncated:
            with pytest.raises(IOError) as error:
                load_one(str(fn_truncated))
    assert str(error.value).endswith(
        "Section <Full Virial Ratio, -(V - W)/T> is not closed at end of file.")


def test_load_one_h2o():
    """Test load_one with h2o sto-3g WFX input."""
    with path('iodata.test.data', 'water_sto3g_hf.wfx') as file_wfx:
        mol = load_one(str(file_wfx))
    assert_allclose(mol.atcoords,
                    np.array([[0.00000000e+00, 0.00000000e+00, 2.40242907e-01],
                              [0.00000000e+00, 1.43244242e+00, -9.60971627e-01],
                              [-1.75417809e-16, -1.43244242e+00, -9.60971627e-01]]),
                    rtol=0., atol=1.e-6)
    assert_allclose(mol.atgradient,
                    np.array([[6.09070231e-16, -5.55187875e-16, -2.29270172e-04],
                              [-2.46849911e-16, -1.18355659e-04, 1.14635086e-04],
                              [-3.62220320e-16, 1.18355659e-04, 1.14635086e-04]]),
                    rtol=0, atol=1.e-6)
    assert_equal(mol.atnums, np.array([8, 1, 1]))
    assert mol.mo.coeffs.shape == (21, 5)
    assert_allclose(mol.energy, -74.965901170787, rtol=0, atol=1.e-6)
    assert mol.extra['keywords'] == 'GTO'
    assert mol.extra['virial_ratio'] == 2.00599838291596
    assert mol.mo.kind == "restricted"
    assert_allclose(mol.mo.energies,
                    np.array([-20.2515479, -1.25760928,
                              -0.59394112, -0.45972872,
                              -0.39261846]), rtol=0, atol=1.e-6)
    assert_equal(mol.mo.occs, np.array([2., 2., 2., 2., 2.]))
    assert_equal(mol.mo.occsa, np.array([1., 1., 1., 1., 1.]))
    assert mol.mo.spinpol == 0.0
    assert mol.mo.nbasis == 21
    assert mol.obasis.nbasis == 21
    assert mol.obasis.primitive_normalization == 'L2'
    assert [shell.icenter for shell in mol.obasis.shells] == [0] * 9 + [1] * 3 + [2] * 3
    assert [shell.kinds for shell in mol.obasis.shells] == [['c']] * 15
    assert_allclose([shell.exponents for shell in mol.obasis.shells[:6]],
                    [[130.709321], [23.8088661], [6.44360831], [5.03315132],
                     [1.16959612], [0.38038896]])
    assert_allclose([shell.exponents for shell in mol.obasis.shells[6:9]],
                    [[5.03315132], [1.16959612], [0.38038896]])
    assert_allclose([shell.exponents for shell in mol.obasis.shells[9:15]],
                    [[3.42525091], [0.62391373], [0.168855404],
                     [3.42525091], [0.62391373], [0.168855404]])
    assert_allclose([shell.coeffs for shell in mol.obasis.shells], [[[1]]] * 15)
    assert mol.obasis_name is None
    assert mol.title == 'H2O HF/STO-3G//HF/STO-3G'
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1.e-5)


def test_load_one_h2():
    """Test load_one with h2 ub3lyp_ccpvtz WFX input."""
    with path('iodata.test.data', 'h2_ub3lyp_ccpvtz.wfx') as file_wfx:
        mol = load_one(str(file_wfx))
    assert_allclose(mol.atcoords,
                    np.array([[0.0, 0.0, 0.7019452462164],
                              [0.0, 0.0, -0.7019452462164]]),
                    rtol=0, atol=1.e-6)
    assert_allclose(mol.atgradient,
                    np.array([[9.74438416e-17, -2.08884441e-16, -7.18565768e-09],
                              [-9.74438416e-17, 2.08884441e-16, 7.18565768e-09]]),
                    rtol=0, atol=1.e-6)
    assert_equal(mol.atnums, np.array([1, 1]))
    assert_allclose(mol.energy, -1.179998789924, rtol=0, atol=1.e-6)
    assert mol.extra['keywords'] == 'GTO'
    assert mol.extra['num_perturbations'] == 0
    assert mol.mo.coeffs.shape == (34, 56)
    assert_allclose(mol.mo.energies[:7],
                    np.array([-0.43408309, 0.0581059, 0.19574763, 0.4705944,
                              0.51160035, 0.51160035, 0.91096805]), rtol=0, atol=1.e-6)
    assert mol.mo.occs.sum() == 2.0
    assert mol.mo.occsa.sum() == 1.0
    assert mol.mo.spinpol == 0.0
    assert mol.mo.nbasis == 34
    assert mol.mo.kind == "unrestricted"
    assert mol.obasis.nbasis == 34
    assert mol.obasis.primitive_normalization == 'L2'
    assert [shell.icenter for shell in mol.obasis.shells] == [0] * 8 + [1] * 8
    assert [shell.kinds for shell in mol.obasis.shells] == [['c']] * 16
    assert_allclose([shell.exponents for shell in mol.obasis.shells],
                    2 * [[33.87], [5.095], [1.159], [0.3258], [0.1027], [1.407], [0.388], [1.057]])
    assert_allclose([shell.coeffs for shell in mol.obasis.shells], [[[1]]] * 16)
    assert mol.obasis_name is None
    assert mol.title == 'h2 ub3lyp/cc-pvtz opt-stable-freq'
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)


def test_load_one_lih_cation_cisd():
    with path('iodata.test.data', 'lih_cation_cisd.wfx') as file_wfx:
        mol = load_one(str(file_wfx))
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == 'unrestricted'
    assert mol.mo.norba == 11
    assert mol.mo.norbb == 11
    assert mol.mo.norb == 22
    assert_allclose(mol.mo.occsa, [
        9.99999999804784E-1, 9.99999998539235E-1, 2.26431690664012E-10,
        5.38480519435475E-11, 0.0, 0.0, 0.0, 0.0, -5.97822590358596E-13,
        -6.01428138426375E-12, -9.12834417514434E-12])
    assert_allclose(mol.mo.occsb, [
        9.99999997403326E-1, 6.03380142010587E-11, 4.36865874834240E-12,
        4.14106040987552E-13, 0.0, 0.0, 0.0, 0.0, -5.69902692731031E-13,
        -1.60216470544366E-11, -3.25430470734432E-10,
    ])
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)


def test_load_one_lih_cation_uhf():
    with path('iodata.test.data', 'lih_cation_uhf.wfx') as file_wfx:
        mol = load_one(str(file_wfx))
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == 'unrestricted'
    assert mol.mo.norba == 2
    assert mol.mo.norbb == 1
    assert mol.mo.norb == 3
    assert_allclose(mol.mo.occsa, [1.0, 1.0])
    assert_allclose(mol.mo.occsb, [1.0])
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)


def test_load_one_lih_cation_rohf():
    with path('iodata.test.data', 'lih_cation_rohf.wfx') as file_wfx:
        mol = load_one(str(file_wfx))
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == 'restricted'
    assert mol.mo.norba == 2
    assert mol.mo.norbb == 2
    assert mol.mo.norb == 2
    assert_allclose(mol.mo.occs, [2.0, 1.0])
    assert_allclose(mol.mo.occsa, [1.0, 1.0])
    assert_allclose(mol.mo.occsb, [1.0, 0.0])
    # check orthonormal mo
    olp = compute_overlap(mol.obasis, mol.atcoords)
    check_orthonormal(mol.mo.coeffsa, olp, 1e-5)
    check_orthonormal(mol.mo.coeffsb, olp, 1e-5)

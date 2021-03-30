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
"""Test iodata.formats.qchemlog module."""

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one
from ..formats.qchemlog import load_qchemlog_low
from ..utils import LineIterator, angstrom, kjmol

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_qchemlog_low_h2o():
    """Test load_qchemlog_low with water_hf_ccpvtz_freq_qchem.out."""
    with path('iodata.test.data', 'water_hf_ccpvtz_freq_qchem.out') as fq:
        data = load_qchemlog_low(LineIterator(str(fq)))

    # check loaded data
    assert data['run_type'] == 'freq'
    assert data['lot'] == 'hf'
    assert data['obasis_name'] == 'cc-pvtz'
    assert data['unrestricted'] == 1
    assert data['symm'] == 0
    assert data['g_rot'] == 1
    assert data['alpha_elec'] == 5
    assert data['beta_elec'] == 5
    assert_allclose(data['nuclear_repulsion_energy'], 9.19775748)
    assert_allclose(data['energy'], -76.0571936393)
    assert data['norba'] == 58
    assert data['norbb'] == 58
    assert data['dipole_tol'] == 2.0231
    assert_allclose(data['enthalpy_dict']['trans_enthalpy'], 0.889)
    assert_allclose(data['enthalpy_dict']['rot_enthalpy'], 0.889)
    assert_allclose(data['enthalpy_dict']['vib_enthalpy'], 13.883)
    assert_allclose(data['enthalpy_dict']['enthalpy_total'], 16.253)
    assert_allclose(data['entropy_dict']['trans_entropy'], 34.608)
    assert_allclose(data['entropy_dict']['rot_entropy'], 11.82)
    assert_allclose(data['entropy_dict']['vib_entropy'], 0.003)
    assert_allclose(data['entropy_dict']['entropy_total'], 46.432)
    assert data['imaginary_freq'] == 0
    assert_allclose(data['vib_energy'], 13.882)
    assert_equal(data['atnums'], np.array([8, 1, 1]))
    assert_equal(data['atmasses'], [15.99491, 1.00783, 1.00783])
    atcoords = np.array([[0.00575, 0.00426, -0.00301],
                         [0.27588, 0.88612, 0.25191],
                         [0.60257, -0.23578, -0.7114]]) * angstrom
    assert_equal(data['atcoords'], atcoords)
    assert_equal(data['mo_a_occ'], np.array([-20.5546, -1.3458, -0.7102, -0.5776, -0.5045]))
    assert_equal(data['mo_b_occ'], np.array([-20.5546, -1.3458, -0.7102, -0.5776, -0.5045]))
    alpha_mo_unoccupied = np.array([0.1423, 0.2041, 0.5445, 0.6021, 0.6682, 0.7874, 0.8014,
                                    0.8052, 0.861, 0.9557, 1.1314, 1.197, 1.5276, 1.5667,
                                    2.0366, 2.052, 2.0664, 2.1712, 2.2342, 2.591, 2.9639,
                                    3.3568, 3.4919, 3.5814, 3.6562, 3.8012, 3.8795, 3.8849,
                                    3.9617, 4.0196, 4.0768, 4.1932, 4.3149, 4.39, 4.5839,
                                    4.6857, 4.8666, 5.1595, 5.2529, 5.5288, 6.0522, 6.5707,
                                    6.9264, 6.9442, 7.0027, 7.0224, 7.068, 7.1668, 7.2377,
                                    7.4574, 7.7953, 8.2906, 12.8843])
    assert_allclose(data['mo_a_vir'], alpha_mo_unoccupied)
    beta_mo_unoccupied = np.array([0.1423, 0.2041, 0.5445, 0.6021, 0.6682, 0.7874, 0.8014,
                                   0.8052, 0.861, 0.9557, 1.1314, 1.197, 1.5276, 1.5667,
                                   2.0366, 2.052, 2.0664, 2.1712, 2.2342, 2.591, 2.9639,
                                   3.3568, 3.4919, 3.5814, 3.6562, 3.8012, 3.8795, 3.8849,
                                   3.9617, 4.0196, 4.0768, 4.1932, 4.3149, 4.39, 4.5839,
                                   4.6857, 4.8666, 5.1595, 5.2529, 5.5288, 6.0522, 6.5707,
                                   6.9264, 6.9442, 7.0027, 7.0224, 7.068, 7.1668, 7.2377,
                                   7.4574, 7.7953, 8.2906, 12.8843])
    assert_allclose(data['mo_b_vir'], beta_mo_unoccupied)
    assert_allclose(data['mulliken_charges'], np.array([-0.482641, 0.241321, 0.241321]))
    assert_allclose(data['dipole'], np.array([1.4989, 1.1097, -0.784]))
    assert_allclose(data['quadrupole'], [-6.1922, 0.2058, -5.0469, -0.9308, 1.1096, -5.762])
    assert_allclose(data['polarizability_tensor'], [[-6.1256608, -0.1911917, 0.8593603],
                                                    [-0.1911917, -7.180854, -1.0224452],
                                                    [0.8593603, -1.0224452, -6.52088]])
    hessian = np.array([[3.162861e-01, 8.366060e-02, -2.326701e-01, -8.253820e-02,
                         -1.226155e-01, -2.676000e-03, -2.337479e-01, 3.895480e-02, 2.353461e-01],
                        [8.366060e-02, 5.460341e-01, 2.252114e-01, -1.647100e-01,
                         -4.652302e-01, -1.071603e-01, 8.104940e-02, -8.080390e-02, -1.180510e-01],
                        [-2.326701e-01, 2.252114e-01, 3.738573e-01, -2.713570e-02,
                         -1.472865e-01, -7.031900e-02, 2.598057e-01, -7.792490e-02, -3.035382e-01],
                        [-8.253820e-02, -1.647100e-01, -2.713570e-02, 7.455040e-02,
                         1.315365e-01, 1.474740e-02, 7.987800e-03, 3.317350e-02, 1.238830e-02],
                        [-1.226155e-01, -4.652302e-01, -1.472865e-01, 1.315365e-01,
                         4.787640e-01, 1.470895e-01, -8.921000e-03, -1.353380e-02, 1.970000e-04],
                        [-2.676000e-03, -1.071603e-01, -7.031900e-02, 1.474740e-02,
                         1.470895e-01, 8.125140e-02, -1.207140e-02, -3.992910e-02, -1.093230e-02],
                        [-2.337479e-01, 8.104940e-02, 2.598057e-01, 7.987800e-03,
                         -8.921000e-03, -1.207140e-02, 2.257601e-01, -7.212840e-02, -2.477343e-01],
                        [3.895480e-02, -8.080390e-02, -7.792490e-02, 3.317350e-02,
                         -1.353380e-02, -3.992910e-02, -7.212840e-02, 9.433770e-02, 1.178541e-01],
                        [2.353461e-01, -1.180510e-01, -3.035382e-01, 1.238830e-02,
                         1.970000e-04, -1.093230e-02, -2.477343e-01, 1.178541e-01, 3.144706e-01]])
    assert_allclose(data['athessian'], hessian)


def test_load_one_qchemlog_freq():
    with path('iodata.test.data', 'water_hf_ccpvtz_freq_qchem.out') as fn_qchemlog:
        mol = load_one(str(fn_qchemlog), fmt='qchemlog')
    assert_allclose(mol.energy, -76.0571936393)
    assert mol.g_rot == 1
    assert mol.nelec == 10
    assert mol.run_type == 'freq'
    assert mol.lot == 'hf'
    assert mol.obasis_name == 'cc-pvtz'
    assert_allclose(mol.atcharges['mulliken'], np.array([-0.482641, 0.241321, 0.241321]))
    assert_equal(mol.moments[(1, 'c')], np.array([1.4989, 1.1097, -0.784]))
    assert_equal(mol.moments[(2, 'c')],
                 np.array([-6.1922, 0.2058, -0.9308, -5.0469, 1.1096, -5.762]))
    hessian = np.array([[3.162861e-01, 8.366060e-02, -2.326701e-01, -8.253820e-02,
                         -1.226155e-01, -2.676000e-03, -2.337479e-01, 3.895480e-02, 2.353461e-01],
                        [8.366060e-02, 5.460341e-01, 2.252114e-01, -1.647100e-01,
                         -4.652302e-01, -1.071603e-01, 8.104940e-02, -8.080390e-02, -1.180510e-01],
                        [-2.326701e-01, 2.252114e-01, 3.738573e-01, -2.713570e-02,
                         -1.472865e-01, -7.031900e-02, 2.598057e-01, -7.792490e-02, -3.035382e-01],
                        [-8.253820e-02, -1.647100e-01, -2.713570e-02, 7.455040e-02,
                         1.315365e-01, 1.474740e-02, 7.987800e-03, 3.317350e-02, 1.238830e-02],
                        [-1.226155e-01, -4.652302e-01, -1.472865e-01, 1.315365e-01,
                         4.787640e-01, 1.470895e-01, -8.921000e-03, -1.353380e-02, 1.970000e-04],
                        [-2.676000e-03, -1.071603e-01, -7.031900e-02, 1.474740e-02,
                         1.470895e-01, 8.125140e-02, -1.207140e-02, -3.992910e-02, -1.093230e-02],
                        [-2.337479e-01, 8.104940e-02, 2.598057e-01, 7.987800e-03,
                         -8.921000e-03, -1.207140e-02, 2.257601e-01, -7.212840e-02, -2.477343e-01],
                        [3.895480e-02, -8.080390e-02, -7.792490e-02, 3.317350e-02,
                         -1.353380e-02, -3.992910e-02, -7.212840e-02, 9.433770e-02, 1.178541e-01],
                        [2.353461e-01, -1.180510e-01, -3.035382e-01, 1.238830e-02,
                         1.970000e-04, -1.093230e-02, -2.477343e-01, 1.178541e-01, 3.144706e-01]])
    assert_equal(mol.athessian, hessian)
    assert mol.extra['nuclear_repulsion_energy'] == 9.19775748
    assert mol.extra['imaginary_freq'] == 0
    # unit conversion for entropy terms, used atomic units + Kalvin
    assert_allclose(mol.extra['entropy_dict']['trans_entropy'], 34.608 * 1.593601437640628e-06)
    assert_allclose(mol.extra['entropy_dict']['rot_entropy'], 11.82 * 1.593601437640628e-06)
    assert_allclose(mol.extra['entropy_dict']['vib_entropy'], 0.003 * 1.593601437640628e-06)
    assert_allclose(mol.extra['entropy_dict']['entropy_total'], 46.432 * 1.593601437640628e-06)
    assert_allclose(mol.extra['vib_energy'], 0.022122375167392933)
    assert_allclose(mol.extra['enthalpy_dict']['trans_enthalpy'], 0.0014167116787071256)
    assert_allclose(mol.extra['enthalpy_dict']['rot_enthalpy'], 0.0014167116787071256)
    assert_allclose(mol.extra['enthalpy_dict']['vib_enthalpy'], 0.022123968768831298)
    assert_allclose(mol.extra['enthalpy_dict']['enthalpy_total'], 0.025900804177758054)
    polarizability_tensor = np.array([[-6.1256608, -0.1911917, 0.8593603],
                                      [-0.1911917, -7.180854, -1.0224452],
                                      [0.8593603, -1.0224452, -6.52088]])
    assert_equal(mol.extra['polarizability_tensor'], polarizability_tensor)
    atcoords = np.array([[0.00575, 0.00426, -0.00301],
                         [0.27588, 0.88612, 0.25191],
                         [0.60257, -0.23578, -0.7114]]) * angstrom
    assert_equal(mol.atcoords, atcoords)
    assert_equal(mol.atmasses, np.array([15.99491, 1.00783, 1.00783]))
    assert_equal(mol.atnums, np.array([8, 1, 1]))
    # molecule orbital related
    # check number of orbitals and occupation numbers
    assert mol.mo.kind == 'unrestricted'
    assert mol.mo.norba == 58
    assert mol.mo.norbb == 58
    assert mol.mo.norb == 116
    assert_equal(mol.mo.occsa, [1, 1, 1, 1, 1] + [0] * 53)
    assert_equal(mol.mo.occsb, [1, 1, 1, 1, 1] + [0] * 53)
    # alpha occupied orbital energies
    occupied = np.array([-20.5546, -1.3458, -0.7102, -0.5776, -0.5045])
    assert_allclose(mol.mo.energies[:5], occupied)
    # beta occupied orbital energies
    assert_allclose(mol.mo.energies[58:63], occupied)
    # alpha virtual orbital energies
    virtual = np.array([0.1423, 0.2041, 0.5445, 0.6021, 0.6682, 0.7874, 0.8014, 0.8052,
                        0.8610, 0.9557, 1.1314, 1.1970, 1.5276, 1.5667, 2.0366, 2.0520,
                        2.0664, 2.1712, 2.2342, 2.5910, 2.9639, 3.3568, 3.4919, 3.5814,
                        3.6562, 3.8012, 3.8795, 3.8849, 3.9617, 4.0196, 4.0768, 4.1932,
                        4.3149, 4.3900, 4.5839, 4.6857, 4.8666, 5.1595, 5.2529, 5.5288,
                        6.0522, 6.5707, 6.9264, 6.9442, 7.0027, 7.0224, 7.0680, 7.1668,
                        7.2377, 7.4574, 7.7953, 8.2906, 12.8843])
    assert_allclose(mol.mo.energies[5:58], virtual)
    # beta virtual orbital energies
    assert_allclose(mol.mo.energies[63:], virtual)


def test_load_qchemlog_low_qchemlog_h2o_dimer_eda2():
    """Test load_qchemlog_low with h2o_dimer_eda_qchem5.3.out."""
    # pylint: disable=too-many-statements
    with path('iodata.test.data', 'h2o_dimer_eda_qchem5.3.out') as fq:
        data = load_qchemlog_low(LineIterator(str(fq)))

    # check loaded data
    assert data['run_type'] == 'eda'
    assert data['lot'] == 'wb97x-v'
    assert data['obasis_name'] == 'def2-tzvpd'
    assert not data['unrestricted']
    assert not data['symm']
    assert data['alpha_elec'] == 10
    assert data['beta_elec'] == 10
    assert_allclose(data['nuclear_repulsion_energy'], 36.66284801)
    assert_allclose(data['energy'], -152.8772543727)
    assert data['norba'] == 116
    assert_allclose(data['dipole_tol'], 2.5701)
    assert_equal(data['atnums'], np.array([8, 1, 1, 8, 1, 1]))
    atcoords = np.array([[-1.5510070000, -0.1145200000, 0.0000000000],
                         [-1.9342590000, 0.7625030000, 0.0000000000],
                         [-0.5996770000, 0.0407120000, 0.0000000000],
                         [1.3506250000, 0.1114690000, 0.0000000000],
                         [1.6803980000, -0.3737410000, -0.7585610000],
                         [1.6803980000, -0.3737410000, 0.7585610000]]) * angstrom
    assert_allclose(data['atcoords'], atcoords)
    assert_allclose(data['mo_a_occ'], np.array([-19.2455, -19.1897, -1.1734, -1.1173, -0.6729,
                                                -0.6242, -0.5373, -0.4825, -0.4530, -0.4045]))

    alpha_mo_unoccupied = np.array([0.0485, 0.0863, 0.0927, 0.1035, 0.1344, 0.1474, 0.1539, 0.1880,
                                    0.1982, 0.2280, 0.2507, 0.2532, 0.2732, 0.2865, 0.2992, 0.3216,
                                    0.3260, 0.3454, 0.3542, 0.3850, 0.3991, 0.4016, 0.4155, 0.4831,
                                    0.5016, 0.5133, 0.5502, 0.5505, 0.5745, 0.5992, 0.6275, 0.6454,
                                    0.6664, 0.6869, 0.7423, 0.7874, 0.8039, 0.8204, 0.8457, 0.9021,
                                    0.9149, 0.9749, 1.0168, 1.0490, 1.1274, 1.2009, 1.6233, 1.6642,
                                    1.6723, 1.6877, 1.7314, 1.7347, 1.8246, 1.8635, 1.8877, 1.9254,
                                    2.0091, 2.1339, 2.2139, 2.2489, 2.2799, 2.3420, 2.3777, 2.5255,
                                    2.6135, 2.6373, 2.6727, 2.7228, 2.8765, 2.8841, 2.9076, 2.9624,
                                    3.0377, 3.0978, 3.2509, 3.3613, 3.8767, 3.9603, 4.0824, 4.1424,
                                    5.1826, 5.2283, 5.3319, 5.3817, 5.4919, 5.5386, 5.5584, 5.5648,
                                    5.6049, 5.6226, 6.1591, 6.2079, 6.3862, 6.4446, 6.5805, 6.5926,
                                    6.6092, 6.6315, 6.6557, 6.6703, 7.0400, 7.1334, 7.1456, 7.2547,
                                    43.7457, 43.9166])
    assert_allclose(data['mo_a_vir'], alpha_mo_unoccupied)
    assert_allclose(data['mulliken_charges'], np.array([-0.610250, 0.304021, 0.337060,
                                                        -0.663525, 0.316347, 0.316347]))
    assert_allclose(data['dipole'], np.array([2.5689, 0.0770, 0.0000]))
    assert_allclose(data['quadrupole'], [-12.0581, -6.2544, -12.8954, -0.0000, -0.0000, -12.2310])
    # check eda2 info
    assert_allclose(data['eda2']['e_elec'], -65.9887)
    assert_allclose(data['eda2']['e_kep_pauli'], 78.5700)
    assert_allclose(data['eda2']['e_disp_free_pauli'], -14.2495)
    assert_allclose(data['eda2']['e_disp'], -7.7384)
    assert_allclose(data['eda2']['e_cls_elec'], -35.1257)
    assert_allclose(data['eda2']['e_cls_pauli'], 25.7192)
    assert_allclose(data['eda2']['e_mod_pauli'], 33.4576)
    assert_allclose(data['eda2']['preparation'], 0.0000)
    assert_allclose(data['eda2']['frozen'], -1.6681)
    assert_allclose(data['eda2']['pauli'], 64.3205)
    assert_allclose(data['eda2']['dispersion'], -7.7384)
    assert_allclose(data['eda2']['polarization'], -4.6371)
    assert_allclose(data['eda2']['charge transfer'], -7.0689)
    assert_allclose(data['eda2']['total'], -21.1126)

    # check fragments
    coords1 = [[-1.551007, -0.11452, 0.0], [-1.934259, 0.762503, 0.0], [-0.599677, 0.040712, 0.0]]
    coords2 = [[1.350625, 0.111469, 0.0], [1.680398, -0.373741, -0.758561],
               [1.680398, -0.373741, 0.758561]]
    assert_equal(len(data['frags']), 2)

    assert_equal(data['frags'][0]['atnums'], [8, 1, 1])
    assert_equal(data['frags'][0]['alpha_elec'], 5)
    assert_equal(data['frags'][0]['beta_elec'], 5)
    assert_equal(data['frags'][0]['nbasis'], 58)
    assert_allclose(data['frags'][0]['atcoords'], np.array(coords1) * angstrom)
    assert_allclose(data['frags'][0]['nuclear_repulsion_energy'], 9.16383018)
    assert_allclose(data['frags'][0]['energy'], -76.4345994141)

    assert_equal(data['frags'][1]['atnums'], [8, 1, 1])
    assert_equal(data['frags'][1]['alpha_elec'], 5)
    assert_equal(data['frags'][1]['beta_elec'], 5)
    assert_equal(data['frags'][1]['nbasis'], 58)
    assert_allclose(data['frags'][1]['atcoords'], np.array(coords2) * angstrom)
    assert_allclose(data['frags'][1]['nuclear_repulsion_energy'], 9.17803894)
    assert_allclose(data['frags'][1]['energy'], -76.4346136883)


def test_load_one_h2o_dimer_eda2():
    """Test load_one with h2o_dimer_eda_qchem5.3.out."""
    # pylint: disable=too-many-statements
    with path('iodata.test.data', 'h2o_dimer_eda_qchem5.3.out') as fn_qchemlog:
        mol = load_one(str(fn_qchemlog), fmt='qchemlog')

    # check loaded data
    assert mol.run_type == 'eda'
    assert mol.lot == 'wb97x-v'
    assert mol.obasis_name == 'def2-tzvpd'
    assert mol.mo.kind == 'restricted'
    # assert not data['symm']
    # # assert data['g_rot'] == 1
    # assert data['alpha_elec'] == 10
    # assert data['beta_elec'] == 10
    assert_allclose(mol.extra['nuclear_repulsion_energy'], 36.66284801)
    assert_allclose(mol.energy, -152.8772543727)
    assert mol.mo.norba == 116
    assert mol.mo.norbb == 116
    assert_equal(mol.atnums, np.array([8, 1, 1, 8, 1, 1]))
    # assert_equal(data['atmasses'], [15.99491, 1.00783, 1.00783])
    atcoords = np.array([[-1.5510070000, -0.1145200000, 0.0000000000],
                         [-1.9342590000, 0.7625030000, 0.0000000000],
                         [-0.5996770000, 0.0407120000, 0.0000000000],
                         [1.3506250000, 0.1114690000, 0.0000000000],
                         [1.6803980000, -0.3737410000, -0.7585610000],
                         [1.6803980000, -0.3737410000, 0.7585610000]]) * angstrom
    assert_equal(mol.atcoords, atcoords)

    # check MO energies
    mo_energies_a = [-19.2455, -19.1897, -1.1734, -1.1173, -0.6729, -0.6242, -0.5373, -0.4825,
                     -0.4530, -0.4045, 0.0485, 0.0863, 0.0927, 0.1035, 0.1344, 0.1474, 0.1539,
                     0.1880, 0.1982, 0.2280, 0.2507, 0.2532, 0.2732, 0.2865, 0.2992, 0.3216,
                     0.3260, 0.3454, 0.3542, 0.3850, 0.3991, 0.4016, 0.4155, 0.4831, 0.5016,
                     0.5133, 0.5502, 0.5505, 0.5745, 0.5992, 0.6275, 0.6454, 0.6664, 0.6869,
                     0.7423, 0.7874, 0.8039, 0.8204, 0.8457, 0.9021, 0.9149, 0.9749, 1.0168,
                     1.0490, 1.1274, 1.2009, 1.6233, 1.6642, 1.6723, 1.6877, 1.7314, 1.7347,
                     1.8246, 1.8635, 1.8877, 1.9254, 2.0091, 2.1339, 2.2139, 2.2489, 2.2799,
                     2.3420, 2.3777, 2.5255, 2.6135, 2.6373, 2.6727, 2.7228, 2.8765, 2.8841,
                     2.9076, 2.9624, 3.0377, 3.0978, 3.2509, 3.3613, 3.8767, 3.9603, 4.0824,
                     4.1424, 5.1826, 5.2283, 5.3319, 5.3817, 5.4919, 5.5386, 5.5584, 5.5648,
                     5.6049, 5.6226, 6.1591, 6.2079, 6.3862, 6.4446, 6.5805, 6.5926, 6.6092,
                     6.6315, 6.6557, 6.6703, 7.0400, 7.1334, 7.1456, 7.2547, 43.7457, 43.9166]
    assert_allclose(mol.mo.energiesa, mo_energies_a)
    assert_allclose(mol.mo.energiesb, mo_energies_a)

    assert_equal(mol.atcharges['mulliken'], np.array([-0.610250, 0.304021, 0.337060,
                                                      -0.663525, 0.316347, 0.316347]))
    assert_allclose(mol.moments[(1, 'c')], np.array([2.5689, 0.0770, 0.0000]))
    assert_allclose(mol.moments[(2, 'c')],
                    [-12.0581, -6.2544, -0.0000, -12.8954, -0.0000, -12.2310])

    # check eda2 info
    assert_equal(mol.extra['eda2']['e_elec'], -65.9887 * kjmol)
    assert_equal(mol.extra['eda2']['e_kep_pauli'], 78.5700 * kjmol)
    assert_equal(mol.extra['eda2']['e_disp_free_pauli'], -14.2495 * kjmol)
    assert_equal(mol.extra['eda2']['e_disp'], -7.7384 * kjmol)
    assert_equal(mol.extra['eda2']['e_cls_elec'], -35.1257 * kjmol)
    assert_equal(mol.extra['eda2']['e_cls_pauli'], 25.7192 * kjmol)
    assert_equal(mol.extra['eda2']['e_mod_pauli'], 33.4576 * kjmol)
    assert_equal(mol.extra['eda2']['preparation'], 0.0000)
    assert_equal(mol.extra['eda2']['frozen'], -1.6681 * kjmol)
    assert_equal(mol.extra['eda2']['pauli'], 64.3205 * kjmol)
    assert_equal(mol.extra['eda2']['dispersion'], -7.7384 * kjmol)
    assert_equal(mol.extra['eda2']['polarization'], -4.6371 * kjmol)
    assert_equal(mol.extra['eda2']['charge transfer'], -7.0689 * kjmol)
    assert_equal(mol.extra['eda2']['total'], -21.1126 * kjmol)

    # check fragments
    coords1 = [[-1.551007, -0.11452, 0.0], [-1.934259, 0.762503, 0.0], [-0.599677, 0.040712, 0.0]]
    coords2 = [[1.350625, 0.111469, 0.0], [1.680398, -0.373741, -0.758561],
               [1.680398, -0.373741, 0.758561]]
    assert_equal(len(mol.extra['frags']), 2)

    assert_equal(mol.extra['frags'][0]['atnums'], [8, 1, 1])
    assert_equal(mol.extra['frags'][0]['alpha_elec'], 5)
    assert_equal(mol.extra['frags'][0]['beta_elec'], 5)
    assert_equal(mol.extra['frags'][0]['nbasis'], 58)
    assert_allclose(mol.extra['frags'][0]['atcoords'], np.array(coords1) * angstrom)
    assert_allclose(mol.extra['frags'][0]['nuclear_repulsion_energy'], 9.16383018)
    assert_allclose(mol.extra['frags'][0]['energy'], -76.4345994141)

    assert_equal(mol.extra['frags'][1]['atnums'], [8, 1, 1])
    assert_equal(mol.extra['frags'][1]['alpha_elec'], 5)
    assert_equal(mol.extra['frags'][1]['beta_elec'], 5)
    assert_equal(mol.extra['frags'][1]['nbasis'], 58)
    assert_allclose(mol.extra['frags'][1]['atcoords'], np.array(coords2) * angstrom)
    assert_allclose(mol.extra['frags'][1]['nuclear_repulsion_energy'], 9.17803894)
    assert_allclose(mol.extra['frags'][1]['energy'], -76.4346136883)

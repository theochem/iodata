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
from ..utils import LineIterator

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def helper_load_data_qchemlog_helper(fn_qchemlog):
    """Load a testing Q-Chem log file with iodata.formats.wfx.load_qchemlog_low."""
    with path('iodata.test.data', fn_qchemlog) as fq:
        lit = LineIterator(str(fq))
        return load_qchemlog_low(lit)


def test_load_data_qchemlog_h2o():
    """Test load_qchemlog_low with water_hf_ccpvtz_freq_qchem.out."""
    data = helper_load_data_qchemlog_helper('water_hf_ccpvtz_freq_qchem.out')
    # check loaded data
    assert data['charge'] == 0
    assert data['natom'] == 3
    assert data['spin_multi'] == 1
    assert data['run_type'] == 'freq'
    assert data['lot'] == 'hf'
    assert data['obasis_name'] == 'cc-pvtz'
    assert data['unrestricted'] == 1
    assert data['symm'] == 0
    assert data['g_rot'] == 1
    assert data['alpha_elec'] == 5
    assert data['beta_elec'] == 5
    assert data['nuclear_repulsion_energy'] == 9.19775748
    assert data['energy'] == -76.0571936393
    assert data['norba'] == 58
    assert data['norbb'] == 58
    assert data['dipole_tol'] == 2.0231
    assert data['enthalpy_dict']['trans_enthalpy'] == 0.889
    assert data['enthalpy_dict']['rot_enthalpy'] == 0.889
    assert data['enthalpy_dict']['vib_enthalpy'] == 13.883
    assert data['enthalpy_dict']['enthalpy_total'] == 16.253
    assert data['entropy_dict']['trans_entropy'] == 34.608
    assert data['entropy_dict']['rot_entropy'] == 11.82
    assert data['entropy_dict']['vib_entropy'] == 0.003
    assert data['entropy_dict']['entropy_total'] == 46.432
    assert data['imaginary_freq'] == 0
    assert data['vib_energy'] == 13.882
    assert_equal(data['atnums'], np.array([8, 1, 1]))
    assert_equal(data['atmasses'], [15.99491, 1.00783, 1.00783])
    atcoords = np.array([[0.00575, 0.00426, -0.00301],
                         [0.27588, 0.88612, 0.25191],
                         [0.60257, -0.23578, -0.7114]])
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
    assert_equal(data['mo_a_vir'], alpha_mo_unoccupied)
    beta_mo_unoccupied = np.array([0.1423, 0.2041, 0.5445, 0.6021, 0.6682, 0.7874, 0.8014,
                                   0.8052, 0.861, 0.9557, 1.1314, 1.197, 1.5276, 1.5667,
                                   2.0366, 2.052, 2.0664, 2.1712, 2.2342, 2.591, 2.9639,
                                   3.3568, 3.4919, 3.5814, 3.6562, 3.8012, 3.8795, 3.8849,
                                   3.9617, 4.0196, 4.0768, 4.1932, 4.3149, 4.39, 4.5839,
                                   4.6857, 4.8666, 5.1595, 5.2529, 5.5288, 6.0522, 6.5707,
                                   6.9264, 6.9442, 7.0027, 7.0224, 7.068, 7.1668, 7.2377,
                                   7.4574, 7.7953, 8.2906, 12.8843])
    assert_equal(data['mo_b_vir'], beta_mo_unoccupied)
    assert_equal(data['mulliken_charges'], np.array([-0.482641, 0.241321, 0.241321]))
    assert_equal(data['dipole'], np.array([1.4989, 1.1097, -0.784]))
    assert_equal(data['quadrupole'],
                 np.array([-6.1922, 0.2058, -5.0469, -0.9308, 1.1096, -5.762]))
    assert_equal(data['polarizability_tensor'], np.array([[-6.1256608, -0.1911917, 0.8593603],
                                                          [-0.1911917, -7.180854, -1.0224452],
                                                          [0.8593603, -1.0224452, -6.52088]]))
    hessian = np.array([[3.162861e-01, 8.366060e-02, -2.326701e-01, -8.253820e-02,
                         -1.226155e-01, -2.676000e-03, -2.337479e-01, 3.895480e-02,
                         2.353461e-01],
                        [8.366060e-02, 5.460341e-01, 2.252114e-01, -1.647100e-01,
                         -4.652302e-01, -1.071603e-01, 8.104940e-02, -8.080390e-02,
                         -1.180510e-01],
                        [-2.326701e-01, 2.252114e-01, 3.738573e-01, -2.713570e-02,
                         -1.472865e-01, -7.031900e-02, 2.598057e-01, -7.792490e-02,
                         -3.035382e-01],
                        [-8.253820e-02, -1.647100e-01, -2.713570e-02, 7.455040e-02,
                         1.315365e-01, 1.474740e-02, 7.987800e-03, 3.317350e-02,
                         1.238830e-02],
                        [-1.226155e-01, -4.652302e-01, -1.472865e-01, 1.315365e-01,
                         4.787640e-01, 1.470895e-01, -8.921000e-03, -1.353380e-02,
                         1.970000e-04],
                        [-2.676000e-03, -1.071603e-01, -7.031900e-02, 1.474740e-02,
                         1.470895e-01, 8.125140e-02, -1.207140e-02, -3.992910e-02,
                         -1.093230e-02],
                        [-2.337479e-01, 8.104940e-02, 2.598057e-01, 7.987800e-03,
                         -8.921000e-03, -1.207140e-02, 2.257601e-01, -7.212840e-02,
                         -2.477343e-01],
                        [3.895480e-02, -8.080390e-02, -7.792490e-02, 3.317350e-02,
                         -1.353380e-02, -3.992910e-02, -7.212840e-02, 9.433770e-02,
                         1.178541e-01],
                        [2.353461e-01, -1.180510e-01, -3.035382e-01, 1.238830e-02,
                         1.970000e-04, -1.093230e-02, -2.477343e-01, 1.178541e-01,
                         3.144706e-01]])
    assert_equal(data['athessian'], hessian)


def test_load_one_qchemlog():
    with path('iodata.test.data', 'water_hf_ccpvtz_freq_qchem.out') as fn_qchemlog:
        mol = load_one(str(fn_qchemlog), fmt='qchemlog')
    assert mol.charge == 0
    assert mol.energy == -76.0571936393
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
                         -1.226155e-01, -2.676000e-03, -2.337479e-01, 3.895480e-02,
                         2.353461e-01],
                        [8.366060e-02, 5.460341e-01, 2.252114e-01, -1.647100e-01,
                         -4.652302e-01, -1.071603e-01, 8.104940e-02, -8.080390e-02,
                         -1.180510e-01],
                        [-2.326701e-01, 2.252114e-01, 3.738573e-01, -2.713570e-02,
                         -1.472865e-01, -7.031900e-02, 2.598057e-01, -7.792490e-02,
                         -3.035382e-01],
                        [-8.253820e-02, -1.647100e-01, -2.713570e-02, 7.455040e-02,
                         1.315365e-01, 1.474740e-02, 7.987800e-03, 3.317350e-02,
                         1.238830e-02],
                        [-1.226155e-01, -4.652302e-01, -1.472865e-01, 1.315365e-01,
                         4.787640e-01, 1.470895e-01, -8.921000e-03, -1.353380e-02,
                         1.970000e-04],
                        [-2.676000e-03, -1.071603e-01, -7.031900e-02, 1.474740e-02,
                         1.470895e-01, 8.125140e-02, -1.207140e-02, -3.992910e-02,
                         -1.093230e-02],
                        [-2.337479e-01, 8.104940e-02, 2.598057e-01, 7.987800e-03,
                         -8.921000e-03, -1.207140e-02, 2.257601e-01, -7.212840e-02,
                         -2.477343e-01],
                        [3.895480e-02, -8.080390e-02, -7.792490e-02, 3.317350e-02,
                         -1.353380e-02, -3.992910e-02, -7.212840e-02, 9.433770e-02,
                         1.178541e-01],
                        [2.353461e-01, -1.180510e-01, -3.035382e-01, 1.238830e-02,
                         1.970000e-04, -1.093230e-02, -2.477343e-01, 1.178541e-01,
                         3.144706e-01]])
    assert_equal(mol.athessian, hessian)
    assert mol.extra['spin_multi'] == 1
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
                         [0.60257, -0.23578, -0.7114]])
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

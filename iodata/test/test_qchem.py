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
"""Test iodata.format.qchem module."""

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from ..formats.qchem import load_qchem_low
from ..api import load_one
from ..utils import LineIterator, angstrom, amu

# from ..utils import LineIterator, angstrom, amu, calorie, avogadro

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_qchem_low_h2o2_hf():
    """Test load_qchem_low() with h2o2 Q-Chem frequency calculation output."""
    with path('iodata.test.data', 'h2o2_hf_sto_3g.freq.out') as fn_qchem:
        lit = LineIterator(str(fn_qchem))
        data = load_qchem_low(lit=lit, lit_hess=None)
    # Unpack data
    title, runtype, basis, exchange, atnames, atomic_num, num_atoms, num_sym, \
        masses, atcoords, polar_matrix, athessian, nucl_repul_energy, \
        num_alpha_electron, num_beta_electron, mol_charges, \
        mulliken_charges, energy, alpha_mo_energy, beta_mo_energy, \
        vib_energy, enthalpy, entropy = data

    assert title == 'H2O2 peroxide  gas phase  Energy minimization'
    assert runtype == 'freq'
    assert basis == 'sto-3g'
    assert exchange == 'HF'
    assert (atnames == ['H', 'O', 'H', 'O']).all()
    assert_allclose(atomic_num, np.array([1, 8, 1, 8]))
    assert_equal(num_atoms, 4)
    assert_equal(num_sym, 2)
    assert_allclose(masses, np.array([i * amu for i in [1.00783, 15.99491, 1.00783, 15.99491]]))
    assert_allclose(atcoords, np.array([[1.96178138, 1.30149218, -0.76479295],
                                        [-1.2960441, 0.24625399, 0.09559935],
                                        [-1.96178138, -1.30149218, -0.76479295],
                                        [1.2960441, -0.24625399, 0.09559935]]))
    assert_allclose(atcoords[0, :], np.array([1.038130, 0.688720, -0.404711]) * angstrom)
    assert_allclose(polar_matrix, np.array([[-9.4103236, -3.1111903, -0.0000000],
                                            [-3.1111903, -5.2719873, -0.0000000],
                                            [-0.0000000, -0.0000000, -1.7166708]]))
    assert_allclose(athessian[0, :], np.array([0.1627798, 0.1488185, -0.0873559,
                                               -0.0379552, -0.0571184, 0.0225419,
                                               -0.0087315, 0.0014284, 0.0048976,
                                               -0.1160929, -0.0931285, 0.0599159]))
    assert_allclose(athessian[:, 2], np.array([-0.0873559, -0.2309311, 0.1271814,
                                               -0.0004128, -0.0076820, 0.0033761,
                                               -0.0048976, -0.0011012, -0.0010603,
                                               0.0926663, 0.2397143, -0.1294975]))
    assert_allclose(nucl_repul_energy, 37.4577990233)
    assert_equal(num_alpha_electron, 9)
    assert_equal(num_beta_electron, 9)
    assert_equal(mol_charges, 0.000000)
    assert_allclose(mulliken_charges, np.array([0.189351, -0.189351, 0.189351, -0.189351]))
    assert_allclose(energy, -148.7649966058)
    assert_allclose(alpha_mo_energy, np.array([-20.317, -20.316, -1.409, -1.104,
                                               -0.598, -0.598, -0.501, -0.393,
                                               -0.349, 0.509, 0.592, 0.615]))
    assert_allclose(beta_mo_energy, np.array([-20.317, -20.316, -1.409, -1.104,
                                              -0.598, -0.598, -0.501, -0.393,
                                              -0.349, 0.509, 0.592, 0.615]))
    assert_allclose(vib_energy, 19.060)
    assert_allclose(enthalpy, 21.802)
    assert_allclose(entropy, 55.347)


# ef test_load_qchem_low_h2o2_hf_hess():
#    """Test load_qchem_low() with h2o2 Q-Chem frequency calculation output."""
#    data = load_qchem_low(filename='iodata/test/data/h2o2_hf_sto_3g.freq.out',
#                          hessfile='iodata/test/data/qchem_hessian.dat')
#    # Unpack data
#    title, basis, exchange, atom_names, atomic_num, num_atoms, num_sym, \
#        masses, coordinates, polar_matrix, hessian, nucl_repul_energy, \
#        num_alpha_electron, num_beta_electron, mol_charges, \
#        mulliken_charges, energy, alpha_mo_energy, beta_mo_energy, \
#        vib_energy, enthalply, entropy = data

#    assert title == 'H2O2 peroxide  gas phase  Energy minimization'
#    assert basis == 'sto-3g'
#    assert exchange == 'HF'
#    assert (atom_names == ['H', 'O', 'H', 'O']).all()
#    assert (atomic_num == [1, 8, 1, 8]).all()
#    assert num_atoms == 4
#    assert num_sym == 2
#    assert (masses == [i * amu for i in [1.00783, 15.99491,
#                                         1.00783, 15.99491]]).all()
#    assert (coordinates[0, :] == np.array([1.038130, 0.688720,
#                                           -0.404711]) * angstrom).all()
#    assert (polar_matrix == np.array(
#        [[-9.4103236, - 3.1111903, - 0.0000000],
#         [- 3.1111903, - 5.2719873, - 0.0000000],
#         [- 0.0000000, - 0.0000000, - 1.7166708]])).all()
#    assert nucl_repul_energy == 37.4577990233
#    assert num_alpha_electron == 9
#    assert num_beta_electron == 9
#    assert mol_charges == 0.000000
#    assert (mulliken_charges == [0.189351, -0.189351,
#                                 0.189351, -0.189351]).all()
#    assert energy == -148.7649966058
#    assert (alpha_mo_energy == [-20.317, -20.316, -1.409, -1.104, -0.598,
#                                -0.598, -0.501, -0.393, -0.349, 0.509,
#                                0.592, 0.615]).all()
#    assert (beta_mo_energy == [-20.317, -20.316, -1.409, -1.104, -0.598,
#                               -0.598, -0.501, -0.393, -0.349, 0.509,
#                               0.592, 0.615]).all()
#    assert vib_energy == 19.060
#    assert enthalply == 21.802
#    assert entropy == 55.347
#    assert hessian.shape == (12, 12)
#    hess_tmp1 = hessian[0, 0] / (1000 * calorie / avogadro / angstrom ** 2)
#    hess_tmp2 = hessian[-1, -1] / (1000 * calorie / avogadro / angstrom ** 2)
#    assert_almost_equal(hess_tmp1, 364.769480916757800060, decimal=7)
#    assert_almost_equal(hess_tmp2, 338.870127396983150447, decimal=7)

def test_load_qchem_h2o2_hf():
    """Test load_one with h2o2 Q-Chem frequency calculation output."""
    with path('iodata.test.data', 'h2o2_hf_sto_3g.freq.out') as fn_qchem:
        mol = load_one(filename=str(fn_qchem), fmt='qchem')

    assert mol.title == 'H2O2 peroxide  gas phase  Energy minimization'
    assert mol.obasis_name == 'sto-3g'
    # assert_equal(mol.atnums, np.array([1, 8, 1, 8]))
    assert_allclose(mol.atcoords, np.array([[1.96178138, 1.30149218, -0.76479295],
                                            [-1.2960441, 0.24625399, 0.09559935],
                                            [-1.96178138, -1.30149218, -0.76479295],
                                            [1.2960441, -0.24625399, 0.09559935]]))
    assert_allclose(mol.athessian[0, :], np.array([0.1627798, 0.1488185, -0.0873559,
                                                   -0.0379552, -0.0571184, 0.0225419,
                                                   -0.0087315, 0.0014284, 0.0048976,
                                                   -0.1160929, -0.0931285, 0.0599159]))
    print(type(mol.athessian))
    assert_equal(mol.charge, 0.000000)
    assert_allclose(mol.energy, -148.7649966058)

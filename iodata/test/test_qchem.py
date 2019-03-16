# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
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
#
# --
# pragma pylint: disable=invalid-name,no-member
"""Test iodata.qchem module."""

import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from ..qchem import load_qchem_low
from ..utils import angstrom, amu, calorie, avogadro

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_qchem_low_h2o2_hf():
    """Test load_qchem_low() with h2o2 Q-Chem frequency calculation output."""
    with path('iodata.test.data', 'h2o2_hf_sto_3g.freq.out') as fn_qchem:
        data = load_qchem_low(filename=fn_qchem, hessfile=None)
    # Unpack data
    title, basis, exchange, atom_names, atomic_num, num_atoms, num_sym, \
        masses, coordinates, polar_matrix, hessian, nucl_repul_energy, \
        num_alpha_electron, num_beta_electron, mol_charges, \
        mulliken_charges, energy, alpha_mo_energy, beta_mo_energy, \
        vib_energy, enthalply, entropy = data

    assert title == 'H2O2 peroxide  gas phase  Energy minimization'
    assert basis == 'sto-3g'
    assert exchange == 'HF'
    assert (atom_names == ['H', 'O', 'H', 'O']).all()
    assert (atomic_num == [1, 8, 1, 8]).all()
    assert num_atoms == 4
    assert num_sym == 2
    assert (masses == [i * amu for i in [1.00783, 15.99491,
                                         1.00783, 15.99491]]).all()
    assert (coordinates[0, :] == np.array([1.038130, 0.688720,
                                           -0.404711]) * angstrom).all()
    assert (polar_matrix == np.array(
        [[-9.4103236, - 3.1111903, - 0.0000000],
         [- 3.1111903, - 5.2719873, - 0.0000000],
         [- 0.0000000, - 0.0000000, - 1.7166708]])).all()
    assert (hessian[0, :] == [0.1627798, 0.1488185, -0.0873559,
                              -0.0379552, -0.0571184, 0.0225419,
                              -0.0087315, 0.0014284, 0.0048976,
                              -0.1160929, -0.0931285, 0.0599159]).all()
    assert (hessian[:, 2] == [-0.0873559, -0.2309311, 0.1271814,
                              -0.0004128, -0.0076820, 0.0033761,
                              -0.0048976, -0.0011012, -0.0010603,
                              0.0926663, 0.2397143, -0.1294975]).all()
    assert nucl_repul_energy == 37.4577990233
    assert num_alpha_electron == 9
    assert num_beta_electron == 9
    assert mol_charges == 0.000000
    assert (mulliken_charges == [0.189351, -0.189351,
                                 0.189351, -0.189351]).all()
    assert energy == -148.7649966058
    assert (alpha_mo_energy == [-20.317, -20.316, -1.409, -1.104, -0.598,
                                -0.598, -0.501, -0.393, -0.349, 0.509,
                                0.592, 0.615]).all()
    assert (beta_mo_energy == [-20.317, -20.316, -1.409, -1.104, -0.598,
                               -0.598, -0.501, -0.393, -0.349, 0.509,
                               0.592, 0.615]).all()
    assert vib_energy == 19.060
    assert enthalply == 21.802
    assert entropy == 55.347


def test_load_qchem_low_h2o2_hf_hess():
    """Test load_qchem_low() with h2o2 Q-Chem frequency calculation output."""
    data = load_qchem_low(filename='iodata/test/data/h2o2_hf_sto_3g.freq.out',
                          hessfile='iodata/test/data/qchem_hessian.dat')
    # Unpack data
    title, basis, exchange, atom_names, atomic_num, num_atoms, num_sym, \
        masses, coordinates, polar_matrix, hessian, nucl_repul_energy, \
        num_alpha_electron, num_beta_electron, mol_charges, \
        mulliken_charges, energy, alpha_mo_energy, beta_mo_energy, \
        vib_energy, enthalply, entropy = data

    assert title == 'H2O2 peroxide  gas phase  Energy minimization'
    assert basis == 'sto-3g'
    assert exchange == 'HF'
    assert (atom_names == ['H', 'O', 'H', 'O']).all()
    assert (atomic_num == [1, 8, 1, 8]).all()
    assert num_atoms == 4
    assert num_sym == 2
    assert (masses == [i * amu for i in [1.00783, 15.99491,
                                         1.00783, 15.99491]]).all()
    assert (coordinates[0, :] == np.array([1.038130, 0.688720,
                                           -0.404711]) * angstrom).all()
    assert (polar_matrix == np.array(
        [[-9.4103236, - 3.1111903, - 0.0000000],
         [- 3.1111903, - 5.2719873, - 0.0000000],
         [- 0.0000000, - 0.0000000, - 1.7166708]])).all()
    assert nucl_repul_energy == 37.4577990233
    assert num_alpha_electron == 9
    assert num_beta_electron == 9
    assert mol_charges == 0.000000
    assert (mulliken_charges == [0.189351, -0.189351,
                                 0.189351, -0.189351]).all()
    assert energy == -148.7649966058
    assert (alpha_mo_energy == [-20.317, -20.316, -1.409, -1.104, -0.598,
                                -0.598, -0.501, -0.393, -0.349, 0.509,
                                0.592, 0.615]).all()
    assert (beta_mo_energy == [-20.317, -20.316, -1.409, -1.104, -0.598,
                               -0.598, -0.501, -0.393, -0.349, 0.509,
                               0.592, 0.615]).all()
    assert vib_energy == 19.060
    assert enthalply == 21.802
    assert entropy == 55.347
    assert hessian.shape == (12, 12)
    hess_tmp1 = hessian[0, 0] / (1000 * calorie / avogadro / angstrom ** 2)
    hess_tmp2 = hessian[-1, -1] / (1000 * calorie / avogadro / angstrom ** 2)
    assert_almost_equal(hess_tmp1, 364.769480916757800060, decimal=7)
    assert_almost_equal(hess_tmp2, 338.870127396983150447, decimal=7)

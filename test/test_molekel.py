# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2016 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import numpy as np

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import

from horton.io.test.common import compute_mulliken_charges, compute_hf_energy


def test_load_mkl_ethanol():
    fn_mkl = context.get_fn('test/ethanol.mkl')
    mol = IOData.from_file(fn_mkl)

    # Direct checks with mkl file
    assert mol.numbers.shape == (9,)
    assert mol.numbers[0] == 1
    assert mol.numbers[4] == 6
    assert mol.coordinates.shape == (9,3)
    assert abs(mol.coordinates[2,1]/angstrom - 2.239037) < 1e-5
    assert abs(mol.coordinates[5,2]/angstrom - 0.948420) < 1e-5
    assert mol.obasis.nbasis == 39
    assert mol.obasis.alphas[0] == 18.731137000
    assert mol.obasis.alphas[10] == 7.868272400
    assert mol.obasis.alphas[-3] == 2.825393700
    #assert mol.obasis.con_coeffs[5] == 0.989450608
    #assert mol.obasis.con_coeffs[7] == 2.079187061
    #assert mol.obasis.con_coeffs[-1] == 0.181380684
    assert (mol.obasis.shell_map[:5] == [0, 0, 1, 1, 1]).all()
    assert (mol.obasis.shell_types[:5] == [0, 0, 0, 0, 1]).all()
    assert (mol.obasis.nprims[-5:] == [3, 1, 1, 3, 1]).all()
    assert mol.exp_alpha.coeffs.shape == (39,39)
    assert mol.exp_alpha.energies.shape == (39,)
    assert mol.exp_alpha.occupations.shape == (39,)
    assert (mol.exp_alpha.occupations[:13] == 1.0).all()
    assert (mol.exp_alpha.occupations[13:] == 0.0).all()
    assert mol.exp_alpha.energies[4] == -1.0206976
    assert mol.exp_alpha.energies[-1] == 2.0748685
    assert mol.exp_alpha.coeffs[0,0] == 0.0000119
    assert mol.exp_alpha.coeffs[1,0] == -0.0003216
    assert mol.exp_alpha.coeffs[-1,-1] == -0.1424743

    # Comparison of derived properties with ORCA output file

    # nuclear-nuclear repulsion
    assert abs(compute_nucnuc(mol.coordinates, mol.pseudo_numbers) - 81.87080034) < 1e-5

    # Check normalization
    olp = mol.obasis.compute_overlap(mol.lf)
    mol.exp_alpha.check_normalization(olp, 1e-5)

    # Mulliken charges
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    expected_charges = np.array([
        0.143316, -0.445861, 0.173045, 0.173021, 0.024542, 0.143066, 0.143080,
        -0.754230, 0.400021
    ])
    assert abs(charges - expected_charges).max() < 1e-5

    # Compute HF energy
    assert abs(compute_hf_energy(mol) - -154.01322894) < 1e-4


def test_load_mkl_li2():
    fn_mkl = context.get_fn('test/li2.mkl')
    mol = IOData.from_file(fn_mkl)

    # Check normalization
    olp = mol.obasis.compute_overlap(mol.lf)
    mol.exp_alpha.check_normalization(olp, 1e-5)
    mol.exp_beta.check_normalization(olp, 1e-5)

    # Check charges
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    expected_charges = np.array([0.5, 0.5])
    assert abs(charges - expected_charges).max() < 1e-5


def test_load_mkl_h2():
    fn_mkl = context.get_fn('test/h2_sto3g.mkl')
    mol = IOData.from_file(fn_mkl)
    olp = mol.obasis.compute_overlap(mol.lf)
    mol.exp_alpha.check_normalization(olp, 1e-5)

    # Compute HF energy
    assert abs(compute_hf_energy(mol) - -1.11750589) < 1e-4

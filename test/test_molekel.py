# -*- coding: utf-8 -*-
# Horton is a development platform for electronic structure methods.
# Copyright (C) 2011-2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>
#
# This file is part of Horton.
#
# Horton is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Horton is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
#pylint: skip-file


import numpy as np

from horton import *
from horton.io.test.common import compute_mulliken_charges, compute_hf_energy


def test_load_mkl_ethanol():
    fn_mkl = context.get_fn('test/ethanol.mkl')
    data = load_smart(fn_mkl)

    # Direct checks with mkl file
    numbers = data['numbers']
    assert numbers.shape == (9,)
    assert numbers[0] == 1
    assert numbers[4] == 6
    coordinates = data['coordinates']
    assert coordinates.shape == (9,3)
    assert abs(coordinates[2,1]/angstrom - 2.239037) < 1e-5
    assert abs(coordinates[5,2]/angstrom - 0.948420) < 1e-5
    obasis = data['obasis']
    assert obasis.nbasis == 39
    assert obasis.alphas[0] == 18.731137000
    assert obasis.alphas[10] == 7.868272400
    assert obasis.alphas[-3] == 2.825393700
    #assert obasis.con_coeffs[5] == 0.989450608
    #assert obasis.con_coeffs[7] == 2.079187061
    #assert obasis.con_coeffs[-1] == 0.181380684
    assert (obasis.shell_map[:5] == [0, 0, 1, 1, 1]).all()
    assert (obasis.shell_types[:5] == [0, 0, 0, 0, 1]).all()
    assert (obasis.nprims[-5:] == [3, 1, 1, 3, 1]).all()
    wfn = data['wfn']
    assert wfn.exp_alpha.coeffs.shape == (39,39)
    assert wfn.exp_alpha.energies.shape == (39,)
    assert wfn.exp_alpha.occupations.shape == (39,)
    assert (wfn.exp_alpha.occupations[:13] == 1.0).all()
    assert (wfn.exp_alpha.occupations[13:] == 0.0).all()
    assert wfn.exp_alpha.energies[4] == -1.0206976
    assert wfn.exp_alpha.energies[-1] == 2.0748685
    assert wfn.exp_alpha.coeffs[0,0] == 0.0000119
    assert wfn.exp_alpha.coeffs[1,0] == -0.0003216
    assert wfn.exp_alpha.coeffs[-1,-1] == -0.1424743

    # Comparison of derived properties with ORCA output file

    # nuclear-nuclear repulsion
    assert abs(compute_nucnuc(coordinates, numbers) - 81.87080034) < 1e-5

    # Check normalization
    lf = data['lf']
    olp = lf.create_one_body()
    obasis.compute_overlap(olp)
    wfn.exp_alpha.check_normalization(olp, 1e-5)

    # Mulliken charges
    charges = compute_mulliken_charges(obasis, lf, numbers, wfn)
    expected_charges = np.array([
        0.143316, -0.445861, 0.173045, 0.173021, 0.024542, 0.143066, 0.143080,
        -0.754230, 0.400021
    ])
    assert abs(charges - expected_charges).max() < 1e-5

    # Compute HF energy
    assert abs(compute_hf_energy(data) - -154.01322894) < 1e-4


def test_load_mkl_li2():
    fn_mkl = context.get_fn('test/li2.mkl')
    data = load_smart(fn_mkl)

    # Check normalization
    wfn = data['wfn']
    lf = data['lf']
    obasis = data['obasis']
    numbers = data['numbers']
    olp = lf.create_one_body()
    obasis.compute_overlap(olp)
    wfn.exp_alpha.check_normalization(olp, 1e-5)
    wfn.exp_beta.check_normalization(olp, 1e-5)

    # Check charges
    charges = compute_mulliken_charges(obasis, lf, numbers, wfn)
    expected_charges = np.array([0.5, 0.5])
    assert abs(charges - expected_charges).max() < 1e-5


def test_load_mkl_h2():
    fn_mkl = context.get_fn('test/h2_sto3g.mkl')
    data = load_smart(fn_mkl)
    wfn = data['wfn']
    lf = data['lf']
    obasis = data['obasis']
    olp = lf.create_one_body()
    obasis.compute_overlap(olp)
    wfn.exp_alpha.check_normalization(olp, 1e-5)

    # Compute HF energy
    assert abs(compute_hf_energy(data) - -1.11750589) < 1e-4

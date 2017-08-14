# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
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


from . common import get_fn, check_normalization
from .. iodata import IOData
from .. utils import angstrom, shells_to_nbasis
from .. overlap import compute_overlap


#TODO: optional gbasis import?


def test_load_mkl_ethanol():
    fn_mkl = get_fn('ethanol.mkl')
    mol = IOData.from_file(fn_mkl)

    # Direct checks with mkl file
    assert mol.numbers.shape == (9,)
    assert mol.numbers[0] == 1
    assert mol.numbers[4] == 6
    assert mol.coordinates.shape == (9, 3)
    assert abs(mol.coordinates[2, 1] / angstrom - 2.239037) < 1e-5
    assert abs(mol.coordinates[5, 2] / angstrom - 0.948420) < 1e-5
    assert shells_to_nbasis(mol.obasis["shell_types"]) == 39
    assert mol.obasis['alphas'][0] == 18.731137000
    assert mol.obasis['alphas'][10] == 7.868272400
    assert mol.obasis['alphas'][-3] == 2.825393700
    # assert mol.obasis.con_coeffs[5] == 0.989450608
    # assert mol.obasis.con_coeffs[7] == 2.079187061
    # assert mol.obasis.con_coeffs[-1] == 0.181380684
    assert (mol.obasis["shell_map"][:5] == [0, 0, 1, 1, 1]).all()
    assert (mol.obasis["shell_types"][:5] == [0, 0, 0, 0, 1]).all()
    assert (mol.obasis['nprims'][-5:] == [3, 1, 1, 3, 1]).all()
    assert mol.orb_alpha_coeffs.shape == (39, 39)
    assert mol.orb_alpha_energies.shape == (39,)
    assert mol.orb_alpha_occs.shape == (39,)
    assert (mol.orb_alpha_occs[:13] == 1.0).all()
    assert (mol.orb_alpha_occs[13:] == 0.0).all()
    assert mol.orb_alpha_energies[4] == -1.0206976
    assert mol.orb_alpha_energies[-1] == 2.0748685
    assert mol.orb_alpha_coeffs[0, 0] == 0.0000119
    assert mol.orb_alpha_coeffs[1, 0] == -0.0003216
    assert mol.orb_alpha_coeffs[-1, -1] == -0.1424743


def test_load_mkl_li2():
    fn_mkl = get_fn('li2.mkl')
    mol = IOData.from_file(fn_mkl)

    # Check normalization
    olp = compute_overlap(**mol.obasis)
    check_normalization(mol.orb_alpha_coeffs, mol.orb_alpha_occs, olp, 1e-5)
    check_normalization(mol.orb_beta_coeffs, mol.orb_beta_occs, olp, 1e-5)


def test_load_mkl_h2():
    fn_mkl = get_fn('h2_sto3g.mkl')
    mol = IOData.from_file(fn_mkl)
    olp = compute_overlap(**mol.obasis)
    check_normalization(mol.orb_alpha_coeffs, mol.orb_alpha_occs, olp, 1e-5)

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


import numpy as np, os

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import

from horton.io.test.common import compute_mulliken_charges
from horton.test.common import tmpdir, compare_mols


def test_load_molden_li2_orca():
    fn_molden = context.get_fn('test/li2.molden.input')
    mol = IOData.from_file(fn_molden)

    # Checkt title
    assert mol.title == 'Molden file created by orca_2mkl for BaseName=li2'

    # Check normalization
    olp = mol.obasis.compute_overlap(mol.lf)
    mol.exp_alpha.check_normalization(olp, 1e-5)
    mol.exp_beta.check_normalization(olp, 1e-5)

    # Check Mulliken charges
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    expected_charges = np.array([0.5, 0.5])
    assert abs(charges - expected_charges).max() < 1e-5


def test_load_molden_h2o_orca():
    fn_molden = context.get_fn('test/h2o.molden.input')
    mol = IOData.from_file(fn_molden)

    # Checkt title
    assert mol.title == 'Molden file created by orca_2mkl for BaseName=h2o'

    # Check normalization
    olp = mol.obasis.compute_overlap(mol.lf)
    mol.exp_alpha.check_normalization(olp, 1e-5)

    # Check Mulliken charges
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    expected_charges = np.array([-0.816308, 0.408154, 0.408154])
    assert abs(charges - expected_charges).max() < 1e-5


def test_load_molden_nh3_molden_pure():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    fn_molden = context.get_fn('test/nh3_molden_pure.molden')
    mol = IOData.from_file(fn_molden)
    # Check Mulliken charges. Comparison with numbers from the Molden program
    # output.
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert abs(charges - molden_charges).max() < 1e-3


def test_load_molden_nh3_molden_cart():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    fn_molden = context.get_fn('test/nh3_molden_cart.molden')
    mol = IOData.from_file(fn_molden)
    # Check Mulliken charges. Comparison with numbers from the Molden program
    # output.
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    print charges
    molden_charges = np.array([0.3138, -0.4300, -0.0667, 0.1829])
    assert abs(charges - molden_charges).max() < 1e-3


def test_load_molden_nh3_orca():
    # The file tested here is created with ORCA. It should be read in
    # properly by altering normalization and sign conventions.
    fn_molden = context.get_fn('test/nh3_orca.molden')
    mol = IOData.from_file(fn_molden)
    # Check Mulliken charges. Comparison with numbers from the Molden program
    # output.
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert abs(charges - molden_charges).max() < 1e-3


def test_load_molden_nh3_psi4():
    # The file tested here is created with PSI4. It should be read in
    # properly by altering normalization conventions.
    fn_molden = context.get_fn('test/nh3_psi4.molden')
    mol = IOData.from_file(fn_molden)
    # Check Mulliken charges. Comparison with numbers from the Molden program
    # output.
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert abs(charges - molden_charges).max() < 1e-3


def test_load_molden_nh3_molpro2012():
    # The file tested here is created with MOLPRO2012.
    fn_molden = context.get_fn('test/nh3_molpro2012.molden')
    mol = IOData.from_file(fn_molden)
    # Check Mulliken charges. Comparison with numbers from the Molden program
    # output.
    dm_full = mol.get_dm_full()
    charges = compute_mulliken_charges(mol.obasis, mol.lf, mol.numbers, dm_full)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert abs(charges - molden_charges).max() < 1e-3


def check_load_dump_consistency(fn):
    mol1 = IOData.from_file(context.get_fn(os.path.join('test', fn)))
    with tmpdir('horton.io.test.test_molden.check_load_dump_consistency.%s' % fn) as dn:
        fn_tmp = os.path.join(dn, 'foo.molden')
        mol1.to_file(fn_tmp)
        mol2 = IOData.from_file(fn_tmp)
    compare_mols(mol1, mol2)


def test_load_dump_consistency_h2o():
    check_load_dump_consistency('h2o.molden.input')


def test_load_dump_consistency_li2():
    check_load_dump_consistency('li2.molden.input')


def test_load_dump_consistency_f():
    check_load_dump_consistency('F.molden')


def test_load_dump_consistency_nh3_molden_pure():
    check_load_dump_consistency('nh3_molden_pure.molden')


def test_load_dump_consistency_nh3_molden_cart():
    check_load_dump_consistency('nh3_molden_cart.molden')

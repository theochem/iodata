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

from .common import get_fn, tmpdir
from ..iodata import IOData


def test_load_fcidump_psi4_h2():
    mol = IOData.from_file(get_fn('FCIDUMP.psi4.h2'))
    assert mol.core_energy == 0.7151043364864863E+00
    assert mol.nelec == 2
    assert mol.ms2 == 0
    assert mol.one_mo.shape == (10, 10)
    assert mol.one_mo[0, 0] == -0.1251399119550580E+01
    assert mol.one_mo[2, 1] == 0.9292454365115077E-01
    assert mol.one_mo[1, 2] == 0.9292454365115077E-01
    assert mol.one_mo[9, 9] == 0.9035054979531029E+00
    assert mol.two_mo.shape == (10, 10, 10, 10)
    assert mol.two_mo[0, 0, 0, 0] == 0.6589928924251115E+00
    # Check physicist's notation and symmetry
    assert mol.two_mo[6, 1, 5, 0] == 0.5335846565304321E-01
    assert mol.two_mo[5, 1, 6, 0] == 0.5335846565304321E-01
    assert mol.two_mo[6, 0, 5, 1] == 0.5335846565304321E-01
    assert mol.two_mo[5, 0, 6, 1] == 0.5335846565304321E-01
    assert mol.two_mo[1, 6, 0, 5] == 0.5335846565304321E-01
    assert mol.two_mo[1, 5, 0, 6] == 0.5335846565304321E-01
    assert mol.two_mo[0, 6, 1, 5] == 0.5335846565304321E-01
    assert mol.two_mo[0, 5, 1, 6] == 0.5335846565304321E-01
    assert mol.two_mo[9, 9, 9, 9] == 0.6273759381091796E+00


def test_load_fcidump_molpro_h2():
    mol = IOData.from_file(get_fn('FCIDUMP.molpro.h2'))
    assert mol.core_energy == 0.7151043364864863E+00
    assert mol.nelec == 2
    assert mol.ms2 == 0
    assert mol.one_mo.shape == (4, 4)
    assert mol.one_mo[0, 0] == -0.1245406261597530E+01
    assert mol.one_mo[0, 1] == -0.1666402467335385E+00
    assert mol.one_mo[1, 0] == -0.1666402467335385E+00
    assert mol.one_mo[3, 3] == 0.3216193420753873E+00
    assert mol.two_mo.shape == (4, 4, 4, 4)
    assert mol.two_mo[0, 0, 0, 0] == 0.6527679278914691E+00
    # Check physicist's notation and symmetry
    assert mol.two_mo[3, 0, 2, 1] == 0.7756042287284058E-01
    assert mol.two_mo[2, 0, 3, 1] == 0.7756042287284058E-01
    assert mol.two_mo[3, 1, 2, 0] == 0.7756042287284058E-01
    assert mol.two_mo[2, 1, 3, 0] == 0.7756042287284058E-01
    assert mol.two_mo[0, 3, 1, 2] == 0.7756042287284058E-01
    assert mol.two_mo[0, 2, 1, 3] == 0.7756042287284058E-01
    assert mol.two_mo[1, 3, 0, 2] == 0.7756042287284058E-01
    assert mol.two_mo[1, 2, 0, 3] == 0.7756042287284058E-01
    assert mol.two_mo[3, 3, 3, 3] == 0.7484308847738417E+00


def test_dump_load_fcidimp_consistency_ao():
    # Setup IOData
    mol0 = IOData.from_file(get_fn('water.xyz'))
    mol0.nelec = 10
    mol0.ms2 = 1
    mol0.one_mo = np.load(get_fn("psi4_h2_one.npy"))
    mol0.two_mo = np.load(get_fn("psi4_h2_two.npy"))

    # Dump to a file and load it again
    with tmpdir('io.test.test_molpro.test_dump_load_fcidump_consistency_ao') as dn:
        mol0.to_file('%s/FCIDUMP' % dn)
        mol1 = IOData.from_file('%s/FCIDUMP' % dn)

    # Compare results
    np.testing.assert_equal(mol0.nelec, mol1.nelec)
    np.testing.assert_equal(mol0.ms2, mol1.ms2)
    np.testing.assert_almost_equal(mol0.one_mo, mol1.one_mo)
    np.testing.assert_almost_equal(mol0.two_mo, mol1.two_mo)

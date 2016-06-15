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
from nose.tools import assert_raises

from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import


def test_typecheck():
    m = IOData(coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    assert issubclass(m.coordinates.dtype.type, float)
    assert not hasattr(m, 'numbers')
    m = IOData(numbers=np.array([2, 3]), coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    m = IOData(numbers=np.array([2.0, 3.0]), pseudo_numbers=np.array([1, 1]), coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    assert issubclass(m.numbers.dtype.type, int)
    assert issubclass(m.pseudo_numbers.dtype.type, float)
    assert hasattr(m, 'numbers')
    del m.numbers
    assert not hasattr(m, 'numbers')
    m = IOData(cube_data=np.array([[[1, 2], [2, 3], [3, 2]]]), coordinates=np.array([[1, 2, 3]]))
    with assert_raises(TypeError):
        IOData(coordinates=np.array([[1, 2], [2, 3]]))
    with assert_raises(TypeError):
        IOData(numbers=np.array([[1, 2], [2, 3]]))
    with assert_raises(TypeError):
        IOData(numbers=np.array([2, 3]), pseudo_numbers=np.array([1]))
    with assert_raises(TypeError):
        IOData(numbers=np.array([2, 3]), coordinates=np.array([[1, 2, 3]]))
    with assert_raises(TypeError):
        IOData(cube_data=np.array([[1, 2], [2, 3], [3, 2]]), coordinates=np.array([[1, 2, 3]]))
    with assert_raises(TypeError):
        IOData(cube_data=np.array([1, 2]))


def test_copy():
    fn_fchk = context.get_fn('test/water_sto3g_hf_g03.fchk')
    fn_log = context.get_fn('test/water_sto3g_hf_g03.log')
    mol1 = IOData.from_file(fn_fchk, fn_log)
    mol2 = mol1.copy()
    assert mol1 != mol2
    vars1 = vars(mol1)
    vars2 = vars(mol2)
    assert len(vars1) == len(vars2)
    for key1, value1 in vars1.iteritems():
        assert value1 is vars2[key1]


def test_dm_water_sto3g_hf():
    fn_fchk = context.get_fn('test/water_sto3g_hf_g03.fchk')
    mol = IOData.from_file(fn_fchk)
    dm = mol.get_dm_full()
    assert abs(dm.get_element(0, 0) - 2.10503807) < 1e-7
    assert abs(dm.get_element(0, 1) - -0.439115917) < 1e-7
    assert abs(dm.get_element(1, 1) - 1.93312061) < 1e-7


def test_dm_lih_sto3g_hf():
    fn_fchk = context.get_fn('test/li_h_3-21G_hf_g09.fchk')
    mol = IOData.from_file(fn_fchk)

    dm = mol.get_dm_full()
    assert abs(dm.get_element(0, 0) - 1.96589709) < 1e-7
    assert abs(dm.get_element(0, 1) - 0.122114249) < 1e-7
    assert abs(dm.get_element(1, 1) - 0.0133112081) < 1e-7
    assert abs(dm.get_element(10, 10) - 4.23924688E-01) < 1e-7

    dm = mol.get_dm_spin()
    assert abs(dm.get_element(0, 0) - 1.40210760E-03) < 1e-9
    assert abs(dm.get_element(0, 1) - -2.65370873E-03) < 1e-9
    assert abs(dm.get_element(1, 1) - 5.38701212E-03) < 1e-9
    assert abs(dm.get_element(10, 10) - 4.23889148E-01) < 1e-7


def test_dm_ch3_rohf_g03():
    fn_fchk = context.get_fn('test/ch3_rohf_sto3g_g03.fchk')
    mol = IOData.from_file(fn_fchk)

    olp = mol.obasis.compute_overlap(mol.lf)
    dm = mol.get_dm_full()
    assert abs(olp.contract_two('ab,ab', dm) - 9) < 1e-6
    dm = mol.get_dm_spin()
    assert abs(olp.contract_two('ab,ab', dm) - 1) < 1e-6

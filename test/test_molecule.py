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

from horton import *
from nose.tools import assert_raises


def test_typecheck():
    m = Molecule(coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    assert issubclass(m.coordinates.dtype.type, float)
    assert not hasattr(m, 'numbers')
    m = Molecule(numbers=np.array([2, 3]), coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    m = Molecule(numbers=np.array([2.0, 3.0]), pseudo_numbers=np.array([1, 1]), coordinates=np.array([[1, 2, 3], [2, 3, 1]]))
    assert issubclass(m.numbers.dtype.type, int)
    assert issubclass(m.pseudo_numbers.dtype.type, float)
    assert hasattr(m, 'numbers')
    del m.numbers
    assert not hasattr(m, 'numbers')
    m = Molecule(cube_data=np.array([[[1, 2], [2, 3], [3, 2]]]), coordinates=np.array([[1, 2, 3]]))
    with assert_raises(TypeError):
        Molecule(coordinates=np.array([[1, 2], [2, 3]]))
    with assert_raises(TypeError):
        Molecule(numbers=np.array([[1, 2], [2, 3]]))
    with assert_raises(TypeError):
        Molecule(numbers=np.array([2, 3]), pseudo_numbers=np.array([1]))
    with assert_raises(TypeError):
        Molecule(numbers=np.array([2, 3]), coordinates=np.array([[1, 2, 3]]))
    with assert_raises(TypeError):
        Molecule(cube_data=np.array([[1, 2], [2, 3], [3, 2]]), coordinates=np.array([[1, 2, 3]]))
    with assert_raises(TypeError):
        Molecule(cube_data=np.array([1, 2]))


def test_copy():
    fn_fchk = context.get_fn('test/water_sto3g_hf_g03.fchk')
    fn_log = context.get_fn('test/water_sto3g_hf_g03.log')
    mol1 = Molecule.from_file(fn_fchk, fn_log)
    mol2 = mol1.copy()
    assert mol1 != mol2
    vars1 = vars(mol1)
    vars2 = vars(mol2)
    assert len(vars1) == len(vars2)
    for key1, value1 in vars1.iteritems():
        assert value1 is vars2[key1]

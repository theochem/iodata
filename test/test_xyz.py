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
from horton.test.common import tmpdir


def test_load_water_number():
    fn = context.get_fn('test/water_number.xyz')
    data = load_smart(fn)
    check_water(data)


def test_load_water_element():
    fn = context.get_fn('test/water_element.xyz')
    data = load_smart(fn)
    check_water(data)


def check_water(data):
    numbers = data['numbers']
    assert numbers[0] == 1
    assert numbers[1] == 8
    assert numbers[2] == 1
    coordinates = data['coordinates']
    assert abs(np.linalg.norm(coordinates[0] - coordinates[1])/angstrom - 0.96) < 1e-5
    assert abs(np.linalg.norm(coordinates[2] - coordinates[1])/angstrom - 0.96) < 1e-5


def test_load_dump_consistency():
    data0 = load_smart(context.get_fn('test/ch3_hf_sto3g.fchk'))

    with tmpdir('horton.io.test.test_xyz.test_load_dump_consistency') as dn:
        dump_smart('%s/test.xyz' % dn, data0)
        data1 = load_smart('%s/test.xyz' % dn)

    assert (data0['numbers'] == data1['numbers']).all()
    assert abs(data0['coordinates'] - data1['coordinates']).max() < 1e-5

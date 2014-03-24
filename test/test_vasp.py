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
from horton.test.common import get_random_cell, tmpdir



def test_unravel_counter():
    from horton.io.vasp import unravel_counter
    assert unravel_counter(0, [3, 3, 3]) == [0, 0, 0]
    assert unravel_counter(0, [2, 4, 3]) == [0, 0, 0]
    assert unravel_counter(1, [2, 4, 3]) == [1, 0, 0]
    assert unravel_counter(2, [2, 4, 3]) == [0, 1, 0]
    assert unravel_counter(3, [2, 4, 3]) == [1, 1, 0]
    assert unravel_counter(8, [2, 4, 3]) == [0, 0, 1]
    assert unravel_counter(9, [2, 4, 3]) == [1, 0, 1]
    assert unravel_counter(11, [2, 4, 3]) == [1, 1, 1]
    assert unravel_counter(24, [2, 4, 3]) == [0, 0, 0]


def test_load_chgcar_oxygen():
    fn = context.get_fn('test/CHGCAR.oxygen')
    data = load_smart(fn)
    assert (data['numbers'] == [8]).all()
    assert abs(data['cell'].volume - (10*angstrom)**3) < 1e-10
    ugrid = data['grid']
    assert len(ugrid.shape) == 3
    assert (ugrid.shape == 2).all()
    assert abs(ugrid.grid_rvecs - data['cell'].rvecs/2).max() < 1e-10
    assert abs(ugrid.origin).max() < 1e-10
    d = data['cube_data']
    assert abs(d[0,0,0] - 0.78406017013E+04/data['cell'].volume) < 1e-10
    assert abs(d[-1,-1,-1] - 0.10024522914E+04/data['cell'].volume) < 1e-10
    assert abs(d[1,0,0] - 0.76183317989E+04/data['cell'].volume) < 1e-10


def test_load_chgcar_water():
    fn = context.get_fn('test/CHGCAR.water')
    data = load_smart(fn)
    assert (data['numbers'] == [8, 1, 1]).all()
    assert abs(data['coordinates'][1] - np.array([0.074983*15+0.903122*1,  0.903122*15,  0.000000])*angstrom).max() < 1e-10
    assert abs(data['cell'].volume - (15*angstrom)**3) < 1e-10
    ugrid = data['grid']
    assert len(ugrid.shape) == 3
    assert (ugrid.shape == 3).all()
    assert abs(ugrid.grid_rvecs - data['cell'].rvecs/3).max() < 1e-10
    assert abs(ugrid.origin).max() < 1e-10


def test_load_locpot_oxygen():
    fn = context.get_fn('test/LOCPOT.oxygen')
    data = load_smart(fn)
    assert (data['numbers'][0] == [8]).all()
    assert abs(data['cell'].volume - (10*angstrom)**3) < 1e-10
    ugrid = data['grid']
    assert len(ugrid.shape) == 3
    assert (ugrid.shape == [1, 4, 2]).all()
    assert abs(ugrid.origin).max() < 1e-10
    d = data['cube_data']
    assert abs(d[0, 0, 0]/electronvolt - 0.35046350435E+01) < 1e-10
    assert abs(d[0, 1, 0]/electronvolt - 0.213732132354E+01) < 1e-10
    assert abs(d[0, 2, 0]/electronvolt - -.65465465497E+01) < 1e-10
    assert abs(d[0, 2, 1]/electronvolt - -.546876467887E+01) < 1e-10


def test_load_poscar_water():
    fn = context.get_fn('test/POSCAR.water')
    data = load_smart(fn)
    assert (data['numbers'] == [8, 1, 1]).all()
    assert abs(data['coordinates'][1] - np.array([0.074983*15,  0.903122*15,  0.000000])*angstrom).max() < 1e-10
    assert abs(data['cell'].volume - (15*angstrom)**3) < 1e-10


def test_load_dump_consistency():
    data0 = load_smart(context.get_fn('test/water_element.xyz'))
    data0['cell'] = get_random_cell(5.0, 3)

    with tmpdir('horton.io.test.test_vasp.test_load_dump_consistency') as dn:
        dump_smart('%s/POSCAR' % dn, data0)
        data1 = load_smart('%s/POSCAR' % dn)

    assert (data1['numbers'] == [8, 1, 1]).all()
    assert abs(data0['coordinates'][1] - data1['coordinates'][0]).max() < 1e-10
    assert abs(data0['coordinates'][0] - data1['coordinates'][1]).max() < 1e-10
    assert abs(data0['coordinates'][2] - data1['coordinates'][2]).max() < 1e-10
    assert abs(data0['cell'].rvecs - data1['cell'].rvecs).max() < 1e-10

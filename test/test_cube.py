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


def test_load_aelta():
    fn_cube = context.get_fn('test/aelta.cube')
    data = load_smart(fn_cube)

    assert len(data['numbers']) == 72
    assert abs(data['coordinates'][5,0] - 27.275511) < 1e-5
    assert abs(data['coordinates'][-2,2] - 26.460812) < 1e-5
    ugrid = data['grid']
    assert (ugrid.shape == 12).all()
    assert data['cell'].nvec == 3
    rvecs = data['cell'].rvecs
    my_rvecs = np.array([[1.8626, 0.1, 0.0], [0.0, 1.8626, 0.0], [0.0, 0.0, 1.8626]], float)*12
    assert abs(rvecs - my_rvecs).max() < 1e-5
    rvecs = ugrid.grid_rvecs
    my_rvecs = np.array([[1.8626, 0.1, 0.0], [0.0, 1.8626, 0.0], [0.0, 0.0, 1.8626]], float)
    assert abs(rvecs - my_rvecs).max() < 1e-5
    assert abs(ugrid.origin - np.array([0.0, 1.2, 0.0])).max() < 1e-10
    cube_data = data['cube_data']
    assert abs(cube_data[0,0,0] - 9.49232e-06) < 1e-12
    assert abs(cube_data[-1,-1,-1] - 2.09856e-04) < 1e-10
    pn = data['pseudo_numbers']
    assert abs(pn[0] - 1.0) < 1e-10
    assert abs(pn[1] - 0.1) < 1e-10
    assert abs(pn[-2] - 0.2) < 1e-10
    assert abs(pn[-1] - data['numbers'][-1]) < 1e-10


def test_load_dump_load_aelta():
    fn_cube1 = context.get_fn('test/aelta.cube')
    data1 = load_smart(fn_cube1)

    with tmpdir('horton.io.test.test_cube.test_load_dump_load_aelta') as dn:
        fn_cube2 = '%s/%s' % (dn, 'aelta.cube')
        dump_smart(fn_cube2, data1)
        data2 = load_smart(fn_cube2)

        assert abs(data1['coordinates'] - data2['coordinates']).max() < 1e-4
        assert (data1['numbers'] == data2['numbers']).all()
        ugrid1 = data1['grid']
        ugrid2 = data2['grid']
        assert abs(ugrid1.grid_rvecs - ugrid2.grid_rvecs).max() < 1e-4
        assert (ugrid1.shape == ugrid2.shape).all()
        assert abs(data1['cube_data'] - data2['cube_data']).max() < 1e-4
        assert abs(data1['pseudo_numbers'] - data2['pseudo_numbers']).max() < 1e-4

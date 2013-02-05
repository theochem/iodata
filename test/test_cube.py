# -*- coding: utf-8 -*-
# Horton is a Density Functional Theory program.
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


import numpy as np, shutil, tempfile
from horton import *


def test_load_aelta():
    lf = DenseLinalgFactory()
    fn_cube = context.get_fn('test/aelta.cube')
    sys = System.from_file(fn_cube)

    assert sys.natom == 72
    assert abs(sys.coordinates[5,0] - 27.275511) < 1e-5
    assert abs(sys.coordinates[-2,2] - 26.460812) < 1e-5
    ui_grid = sys.props['ui_grid']
    assert (ui_grid.shape == 12).all()
    assert sys.cell.nvec == 3
    rvecs = sys.cell.rvecs
    my_rvecs = np.array([[1.8626, 0.1, 0.0], [0.0, 1.8626, 0.0], [0.0, 0.0, 1.8626]], float)*12
    assert abs(rvecs - my_rvecs).max() < 1e-5
    assert ui_grid.grid_cell.nvec == 3
    rvecs = ui_grid.grid_cell.rvecs
    my_rvecs = np.array([[1.8626, 0.1, 0.0], [0.0, 1.8626, 0.0], [0.0, 0.0, 1.8626]], float)
    assert abs(rvecs - my_rvecs).max() < 1e-5
    assert abs(ui_grid.origin - np.array([0.0, 1.2, 0.0])).max() < 1e-10
    data = sys.props['cube_data']
    assert abs(data[0,0,0] - 9.49232e-06) < 1e-12
    assert abs(data[-1,-1,-1] - 2.09856e-04) < 1e-10
    nc = sys.props['nuclear_charges']
    assert abs(nc[0] - 1.0) < 1e-10
    assert abs(nc[1] - 0.1) < 1e-10
    assert abs(nc[-2] - 0.2) < 1e-10
    assert abs(nc[-1]) < 1e-10


def test_load_dump_load_aelta():
    lf = DenseLinalgFactory()
    fn_cube1 = context.get_fn('test/aelta.cube')
    sys1 = System.from_file(fn_cube1)

    dn = tempfile.mkdtemp('test_load_dump_load_aelta', 'horton.io.test.test_cube')
    try:
        fn_cube2 = '%s/%s' % (dn, 'aelta.cube')
        sys1.to_file(fn_cube2)
        sys2 = System.from_file(fn_cube2)

        assert sys1.natom == sys2.natom
        assert abs(sys1.coordinates - sys2.coordinates).max() < 1e-4
        assert (sys1.numbers == sys2.numbers).all()
        ui_grid1 = sys1.props['ui_grid']
        ui_grid2 = sys2.props['ui_grid']
        assert abs(ui_grid1.grid_cell.rvecs - ui_grid2.grid_cell.rvecs).max() < 1e-4
        assert (ui_grid1.shape == ui_grid2.shape).all()
        assert abs(sys1.props['cube_data'] - sys2.props['cube_data']).max() < 1e-4
        assert abs(sys1.props['nuclear_charges'] - sys2.props['nuclear_charges']).max() < 1e-4
    finally:
        shutil.rmtree(dn)

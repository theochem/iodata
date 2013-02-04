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


import numpy as np
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
    assert ui_grid.cell.nvec == 3
    rvecs = ui_grid.cell.rvecs
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

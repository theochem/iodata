# IODATA is an input and output module for quantum chemistry.
# Copyright (C) 2011-2019 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
# pylint: disable=no-member
"""Test iodata.formats.cube module."""


import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..iodata import load_one, dump_one

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_aelta():
    with path('iodata.test.data', 'aelta.cube') as fn_cube:
        mol = load_one(str(fn_cube))
    assert mol.title == 'Some random cube for testing (sort of) useless data'
    assert_equal(mol.natom, 72)
    assert_allclose(mol.coordinates[5, 0], 27.275511, atol=1.e-5)
    assert_allclose(mol.coordinates[-2, 2], 26.460812, atol=1.e-5)
    assert_equal(mol.grid['shape'], 12)
    rvecs = mol.cell
    my_rvecs = np.array([[1.8626, 0.1, 0.0],
                         [0.0, 1.8626, 0.0],
                         [0.0, 0.0, 1.8626]], dtype=np.float) * 12
    assert_allclose(rvecs, my_rvecs, atol=1.e-5)
    rvecs = mol.grid['grid_rvecs']
    my_rvecs = np.array([[1.8626, 0.1, 0.0],
                         [0.0, 1.8626, 0.0],
                         [0.0, 0.0, 1.8626]], dtype=np.float)
    assert_allclose(rvecs, my_rvecs, atol=1.e-5)
    assert_allclose(mol.grid['origin'], np.array([0.0, 1.2, 0.0]), atol=1.e-10)

    assert_allclose(mol.cube_data[0, 0, 0], 9.49232e-06, atol=1.e-12)
    assert_allclose(mol.cube_data[-1, -1, -1], 2.09856e-04, atol=1.e-10)
    pn = mol.pseudo_numbers
    assert_allclose(pn[0], 1.0, atol=1.e-10)
    assert_allclose(pn[1], 0.1, atol=1.e-10)
    assert_allclose(pn[-2], 0.2, atol=1.e-10)
    assert_allclose(pn[-1], mol.atnums[-1], atol=1.e-10)


def test_load_dump_load_aelta(tmpdir):
    with path('iodata.test.data', 'aelta.cube') as fn_cube1:
        mol1 = load_one(str(fn_cube1))

    fn_cube2 = '%s/%s' % (tmpdir, 'aelta.cube')
    dump_one(mol1, fn_cube2)
    mol2 = load_one(fn_cube2)

    assert mol1.title == mol2.title
    assert_allclose(mol1.coordinates, mol2.coordinates, atol=1.e-4)
    assert_equal(mol1.atnums, mol2.atnums)
    ugrid1 = mol1.grid
    ugrid2 = mol2.grid
    assert_allclose(ugrid1["grid_rvecs"], ugrid2["grid_rvecs"], atol=1.e-4)
    assert_equal(ugrid1["shape"], ugrid2["shape"])
    assert_allclose(mol1.cube_data, mol2.cube_data, atol=1.e-4)
    assert_allclose(mol1.pseudo_numbers, mol2.pseudo_numbers, atol=1.e-4)

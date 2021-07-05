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
# pylint: disable=unsubscriptable-object
"""Test iodata.formats.cube module."""


import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one, dump_one

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_aelta():
    with path('iodata.test.data', 'aelta.cube') as fn_cube:
        mol = load_one(str(fn_cube))
    assert mol.title == 'Some random cube for testing (sort of) useless data'
    assert_equal(mol.natom, 72)
    assert_allclose(mol.atcoords[5, 0], 27.275511, atol=1.e-5)
    assert_allclose(mol.atcoords[-2, 2], 26.460812, atol=1.e-5)
    assert_equal(mol.cube.shape, (12, 12, 12))
    my_cellvecs = np.array([[1.8626, 0.1, 0.0],
                            [0.0, 1.8626, 0.0],
                            [0.0, 0.0, 1.8626]], dtype=np.float) * 12
    assert_allclose(mol.cellvecs, my_cellvecs, atol=1.e-5)
    my_axes = np.array([[1.8626, 0.1, 0.0],
                        [0.0, 1.8626, 0.0],
                        [0.0, 0.0, 1.8626]], dtype=np.float)
    assert_allclose(mol.cube.axes, my_axes, atol=1.e-5)
    assert_allclose(mol.cube.origin, np.array([0.0, 1.2, 0.0]), atol=1.e-10)

    assert_allclose(mol.cube.data[0, 0, 0], 9.49232e-06, atol=1.e-12)
    assert_allclose(mol.cube.data[-1, -1, -1], 2.09856e-04, atol=1.e-10)
    pn = mol.atcorenums
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

    with open(fn_cube2) as f:
        line_counter = 0
        block_counter = 0
        for line in f:
            line_counter += 1
            if line_counter > 6 + len(mol2.atnums):
                if mol2.cube.shape[2] % 6 == 0:
                    assert len(line.split()) == 6
                if mol2.cube.shape[2] % 6 != 0:
                    block_line_counter = line_counter - (
                        6
                        + len(mol2.atnums)
                        + block_counter * (mol2.cube.shape[2] // 6 + 1)
                    )
                    if 1 <= block_line_counter <= mol2.cube.shape[2] // 6:
                        assert len(line.split()) == 6
                    if block_line_counter == mol2.cube.shape[2] // 6 + 1:
                        assert len(line.split()) == mol2.cube.shape[2] % 6

    assert mol1.title == mol2.title
    assert_allclose(mol1.atcoords, mol2.atcoords, atol=1.e-4)
    assert_equal(mol1.atnums, mol2.atnums)
    cube1 = mol1.cube
    cube2 = mol2.cube
    assert_allclose(cube1.axes, cube2.axes, atol=1.e-4)
    assert_equal(cube1.shape, cube2.shape)
    assert_allclose(mol1.cube.data, mol2.cube.data, atol=1.e-4)
    assert_allclose(mol1.atcorenums, mol2.atcorenums, atol=1.e-4)


def test_load_dump_h2o_5points(tmpdir):
    # load cube file
    with path('iodata.test.data', 'cubegen_h2o_5points.cube') as fn_cube1:
        mol1 = load_one(str(fn_cube1))
    # write cube file in a temporary directory
    fn_cube2 = tmpdir.join('iodata_h2o_5points.cube')
    dump_one(mol1, fn_cube2)
    # read the contents as string (skip the first 2 lines) & compare
    with open(fn_cube1, "r") as f:
        content1 = f.read().split("\n", 2)[-1]
    content2 = fn_cube2.read().split("\n", 2)[-1]
    assert content1 == content2


def test_load_dump_ch4_6points(tmpdir):
    # load cube file
    with path('iodata.test.data', 'cubegen_ch4_6points.cube') as fn_cube1:
        mol1 = load_one(str(fn_cube1))
    # write cube file in a temporary directory
    fn_cube2 = tmpdir.join('iodata_ch4_6points.cube')
    dump_one(mol1, fn_cube2)
    # read the contents as string (skip the first 2 lines) & compare
    with open(fn_cube1, "r") as f:
        content1 = f.read().split("\n", 2)[-1]
    content2 = fn_cube2.read().split("\n", 2)[-1]
    assert content1 == content2


def test_load_dump_nh3_7points(tmpdir):
    # load cube file
    with path('iodata.test.data', 'cubegen_nh3_7points.cube') as fn_cube1:
        mol1 = load_one(str(fn_cube1))
    # write cube file in a temporary directory
    fn_cube2 = tmpdir.join('iodata_nh3_7points.cube')
    dump_one(mol1, fn_cube2)
    # read the contents as string (skip the first 2 lines) & compare
    with open(fn_cube1, "r") as f:
        content1 = f.read().split("\n", 2)[-1]
    content2 = fn_cube2.read().split("\n", 2)[-1]
    assert content1 == content2

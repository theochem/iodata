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
# pragma pylint: disable=invalid-name
"""Test iodata.poscar module."""

import os

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..utils import angstrom, volume
from ..iodata import IOData

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_poscar_water():
    with path('iodata.test.data', 'POSCAR.water') as fn:
        mol = IOData.from_file(str(fn))
    assert mol.title == 'Water molecule in a box'
    assert_equal(mol.numbers, [8, 1, 1])
    coords = np.array([0.074983 * 15, 0.903122 * 15, 0.000000])
    assert_allclose(mol.coordinates[1], coords, atol=1e-7)
    assert_allclose(volume(mol.rvecs), 15 ** 3, atol=1.e-4)


def test_load_poscar_cubicbn_cartesian():
    with path('iodata.test.data', 'POSCAR.cubicbn_cartesian') as fn:
        mol = IOData.from_file(str(fn))
    assert mol.title == 'Cubic BN'
    assert_equal(mol.numbers, [5, 7])
    assert_allclose(mol.coordinates[1],
                    np.array([0.25] * 3) * 3.57 * angstrom, atol=1.e-10)
    assert_allclose(volume(mol.rvecs), (3.57 * angstrom) ** 3 / 4, atol=1.e-10)


def test_load_poscar_cubicbn_direct():
    with path('iodata.test.data', 'POSCAR.cubicbn_direct') as fn:
        mol = IOData.from_file(str(fn))
    assert mol.title == 'Cubic BN'
    assert_equal(mol.numbers, [5, 7])
    assert_allclose(mol.coordinates[1],
                    np.array([0.25] * 3) * 3.57 * angstrom, atol=1.e-10)
    assert_allclose(volume(mol.rvecs), (3.57 * angstrom) ** 3 / 4, 1.e-10)


def test_load_dump_consistency(tmpdir):
    with path('iodata.test.data', 'water_element.xyz') as fn:
        mol0 = IOData.from_file(str(fn))
    # random matrix generated from a uniform distribution on [0., 5.0)
    mol0.rvecs = np.array([[2.05278155, 0.23284023, 1.59024118],
                           [4.96430141, 4.73044423, 4.67590975],
                           [3.48374425, 0.67931228, 0.66281160]])
    mol0.gvecs = np.linalg.inv(mol0.rvecs).T

    fn_tmp = os.path.join(tmpdir, 'POSCAR')
    mol0.to_file(fn_tmp)
    mol1 = IOData.from_file(fn_tmp)

    assert mol0.title == mol1.title
    assert_equal(mol1.numbers, [8, 1, 1])
    assert_allclose(mol0.coordinates[1], mol1.coordinates[0], atol=1.e-10)
    assert_allclose(mol0.coordinates[0], mol1.coordinates[1], atol=1.e-10)
    assert_allclose(mol0.coordinates[2], mol1.coordinates[2], atol=1.e-10)
    assert_allclose(mol0.rvecs, mol1.rvecs, atol=1.e-10)

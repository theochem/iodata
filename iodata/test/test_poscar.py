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


import numpy as np

from . common import tmpdir, get_random_cell
from .. utils import angstrom, volume
from .. iodata import IOData
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_poscar_water():
    with path('iodata.test.data', 'POSCAR.water') as fn:
        mol = IOData.from_file(str(fn))
    assert mol.title == 'Water molecule in a box'
    assert (mol.numbers == [8, 1, 1]).all()
    coords = np.array([0.074983 * 15, 0.903122 * 15, 0.000000])
    assert abs(mol.coordinates[1] - coords).max() < 1e-7
    assert abs(volume(mol.rvecs) - 15 ** 3) < 1e-4


def test_load_poscar_cubicbn_cartesian():
    with path('iodata.test.data', 'POSCAR.cubicbn_cartesian') as fn:
        mol = IOData.from_file(str(fn))
    assert mol.title == 'Cubic BN'
    assert (mol.numbers == [5, 7]).all()
    assert abs(mol.coordinates[1] - np.array([0.25] * 3) * 3.57 * angstrom).max() < 1e-10
    assert abs(volume(mol.rvecs) - (3.57 * angstrom) ** 3 / 4) < 1e-10


def test_load_poscar_cubicbn_direct():
    with path('iodata.test.data', 'POSCAR.cubicbn_direct') as fn:
        mol = IOData.from_file(str(fn))
    assert mol.title == 'Cubic BN'
    assert (mol.numbers == [5, 7]).all()
    assert abs(mol.coordinates[1] - np.array([0.25] * 3) * 3.57 * angstrom).max() < 1e-10
    assert abs(volume(mol.rvecs) - (3.57 * angstrom) ** 3 / 4) < 1e-10


def test_load_dump_consistency():
    with path('iodata.test.data', 'water_element.xyz') as fn:
        mol0 = IOData.from_file(str(fn))
    mol0.rvecs = get_random_cell(5.0, 3)
    mol0.gvecs = np.linalg.inv(mol0.rvecs).T

    with tmpdir('io.test.test_vasp.test_load_dump_consistency') as dn:
        mol0.to_file('%s/POSCAR' % dn)
        mol1 = IOData.from_file('%s/POSCAR' % dn)

    assert mol0.title == mol1.title
    assert (mol1.numbers == [8, 1, 1]).all()
    assert abs(mol0.coordinates[1] - mol1.coordinates[0]).max() < 1e-10
    assert abs(mol0.coordinates[0] - mol1.coordinates[1]).max() < 1e-10
    assert abs(mol0.coordinates[2] - mol1.coordinates[2]).max() < 1e-10
    assert abs(mol0.rvecs - mol1.rvecs).max() < 1e-10

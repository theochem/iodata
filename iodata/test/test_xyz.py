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
# pragma pylint: disable=invalid-name,no-member
"""Test iodata.xyz module."""


import numpy as np

from .common import get_fn, tmpdir

from ..iodata import IOData
from ..utils import angstrom


def test_load_water_number():
    # test xyz with atomic numbers
    mol = IOData.from_file(get_fn('water_number.xyz'))
    check_water(mol)


def test_load_water_element():
    # test xyz file with atomic symbols
    mol = IOData.from_file(get_fn('water_element.xyz'))
    check_water(mol)


def check_water(mol):
    assert mol.title == 'Water'
    assert mol.numbers[0] == 1
    assert mol.numbers[1] == 8
    assert mol.numbers[2] == 1
    # check bond length
    print(np.linalg.norm(mol.coordinates[0] - mol.coordinates[2]) / angstrom)
    assert abs(np.linalg.norm(mol.coordinates[0] - mol.coordinates[1]) / angstrom - 0.960) < 1e-5
    assert abs(np.linalg.norm(mol.coordinates[2] - mol.coordinates[1]) / angstrom - 0.960) < 1e-5
    assert abs(np.linalg.norm(mol.coordinates[0] - mol.coordinates[2]) / angstrom - 1.568) < 1e-3


def test_load_dump_consistency():
    mol0 = IOData.from_file(get_fn('ch3_hf_sto3g.fchk'))
    # write xyz file in a temporary folder & then read it
    with tmpdir('io.test.test_xyz.test_load_dump_consistency') as dn:
        mol0.to_file('%s/test.xyz' % dn)
        mol1 = IOData.from_file('%s/test.xyz' % dn)
    # check two xyz files
    assert mol0.title == mol1.title
    assert (mol0.numbers == mol1.numbers).all()
    assert abs(mol0.coordinates - mol1.coordinates).max() < 1e-5


def test_dump_xyz_water_element():
    mol0 = IOData.from_file(get_fn('water_element.xyz'))
    with tmpdir('io.test.test_xyz.test_dump_xyz_water_element') as dn:
        mol0.to_file('%s/test.xyz' % dn)
        mol1 = IOData.from_file('%s/test.xyz' % dn)
    # check two xyz file
    assert mol0.title == mol1.title
    assert (mol0.numbers == mol1.numbers).all()
    assert abs(mol0.coordinates - mol1.coordinates).max() < 1e-5


def test_dump_xyz_water_number():
    mol0 = IOData.from_file(get_fn('water_number.xyz'))
    with tmpdir('io.test.test_xyz.test_dump_xyz_water_number') as dn:
        mol0.to_file('%s/test.xyz' % dn)
        mol1 = IOData.from_file('%s/test.xyz' % dn)
    # check two xyz file
    assert mol0.title == mol1.title
    assert (mol0.numbers == mol1.numbers).all()
    assert abs(mol0.coordinates - mol1.coordinates).max() < 1e-5

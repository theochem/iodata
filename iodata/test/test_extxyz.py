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
"""Test iodata.formats.extxyz module."""

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one, load_many
from ..utils import angstrom
try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_fcc_extended():
    with path('iodata.test.data', 'al_fcc.xyz') as fn_xyz:
        mol = load_one(str(fn_xyz), fmt='extxyz')
    assert hasattr(mol, 'energy')
    assert isinstance(mol.energy, float)
    assert_allclose(mol.energy, -112.846680723)
    assert hasattr(mol, 'cellvecs')
    assert mol.cellvecs.dtype == float
    assert_allclose(mol.cellvecs, np.eye(3) * 7.6 * angstrom)
    assert 'pbc' in mol.extra
    assert mol.extra['pbc'].dtype == bool
    assert_equal(mol.extra['pbc'], np.array([True, False, True]))
    assert 'species' in mol.extra
    assert_equal(mol.extra['species'], np.array(['Al'] * 32))
    assert mol.atgradient.shape == (mol.natom, 3)
    assert_allclose(mol.atgradient[0, 0], -0.285831)
    assert_allclose(mol.atgradient[2, 1], 0.268537)
    assert_allclose(mol.atgradient[-1, -1], -0.928032)
    assert_allclose(mol.atcoords[31, 1], 5.56174271338 * angstrom)


def test_load_mgo():
    with path('iodata.test.data', 'mgo.xyz') as fn_xyz:
        mol = load_one(str(fn_xyz), fmt='extxyz')
    assert_equal(mol.atnums, [12] * 4 + [8] * 4)
    assert_equal(mol.atcoords[3], np.array([1, 1, 0]) * 2.10607000 * angstrom)
    assert_equal(mol.extra["spacegroup"], ["F", "m", "-3", "m"])
    assert mol.extra["unit_cell"] == "conventional"
    assert_equal(mol.extra["pbc"], [True, True, True])
    assert_equal(mol.cellvecs, np.identity(3) * 4.21214 * angstrom)


def test_load_many_extended():
    with path('iodata.test.data', 'water_extended_trajectory.xyz') as fn_xyz:
        mols = list(load_many(str(fn_xyz), fmt='extxyz'))
    assert len(mols) == 3
    assert 'some_label' in mols[0].extra
    assert_equal(mols[0].extra['some_label'],
                 np.array([[True, True], [False, True], [False, False]]))
    assert 'is_true' in mols[0].extra
    assert mols[0].extra['is_true']
    assert hasattr(mols[0], 'charge')
    assert_allclose(mols[0].charge, 0)
    assert 'pi' in mols[1].extra
    assert_equal(mols[1].extra['pi'], 3.14)
    assert_equal(mols[1].atnums, np.array([8, 1, 1, 1]))
    assert_equal(mols[1].atcoords[-1, 2], -12 * angstrom)
    assert_equal(mols[2].atnums, np.array([8, 1, 1]))
    assert hasattr(mols[2], 'atmasses')
    assert mols[2].atmasses.dtype == float
    assert_allclose(mols[2].atmasses, np.array([29164.39290107, 1837.47159474, 1837.47159474]))

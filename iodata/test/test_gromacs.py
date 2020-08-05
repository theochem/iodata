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
"""Test iodata.formats.gromacs module."""

from numpy.testing import assert_equal, assert_allclose

from ..api import load_one, load_many
from ..utils import nanometer, picosecond

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_water():
    # test gro file of one water
    with path('iodata.test.data', 'water.gro') as fn_gro:
        mol = load_one(str(fn_gro))
    check_water(mol)


def check_water(mol):
    """Test some things on a water file."""
    assert mol.title == 'MD of 2 waters'
    assert mol.atcoords.shape == (6, 3)
    assert_allclose(mol.atcoords[-1] / nanometer, [1.326, 0.120, 0.568])
    assert mol.atffparams['attypes'][2] == 'HW3'
    assert mol.atffparams['resnames'][-1] == 'WATER'
    assert_equal(mol.atffparams['resnums'][2:4], [1, 2])
    assert_allclose(mol.cellvecs[0][0], 1.82060 * nanometer, atol=1.e-5)
    assert mol.extra['velocities'].shape == (6, 3)
    vel = mol.extra['velocities'][-1]
    assert_allclose(vel * (picosecond / nanometer), [1.9427, -0.8216, -0.0244])


def test_load_many():
    with path('iodata.test.data', 'water2.gro') as fn_gro:
        mols = list(load_many(str(fn_gro)))
    assert len(mols) == 2
    assert mols[0].extra['time'] == 0.0 * picosecond
    assert mols[1].extra['time'] == 1.0 * picosecond
    for mol in mols:
        check_water(mol)

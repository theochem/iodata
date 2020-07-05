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
"""Test iodata.formats.orcalog module."""

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ..api import load_one
from ..utils import angstrom

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_water_number():
    with path('iodata.test.data', 'water_orca.out') as fn:
        mol = load_one(fn)
    # Test atomic numbers and number of atoms
    assert mol.natom == 3
    assert_equal(mol.atnums, [8, 1, 1])
    # check bond length
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[1]) / angstrom, 0.9500, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.atcoords[0] - mol.atcoords[2]) / angstrom, 0.9500, atol=1.e-5)
    assert_allclose(np.linalg.norm(
        mol.atcoords[1] - mol.atcoords[2]) / angstrom, 1.5513, atol=1.e-4)
    # check energies of scf cycles
    energies = np.array([-76.34739931, -76.34740001, -76.34740005, -76.34740029])
    assert_allclose(mol.extra['scf_energies'], energies)
    # check scf energy
    assert_allclose(mol.energy, -76.347791524303, atol=1e-8)
    # check dipole moment
    assert_allclose(mol.moments[(1, 'c')], [0.76499, 0.00000, 0.54230])

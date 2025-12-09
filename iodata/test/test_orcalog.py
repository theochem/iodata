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
"""Test iodata.formats.orcalog module."""

from importlib.resources import as_file, files

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from ..api import load_one
from ..utils import angstrom


def test_load_water_number():
    with as_file(files("iodata.test.data").joinpath("water_orca.out")) as fn:
        mol = load_one(fn)
    # Test atomic numbers and number of atoms
    assert mol.natom == 3
    assert_equal(mol.atnums, [8, 1, 1])
    # check bond length
    assert_allclose(
        np.linalg.norm(mol.atcoords[0] - mol.atcoords[1]) / angstrom, 0.9500, atol=1.0e-5
    )
    assert_allclose(
        np.linalg.norm(mol.atcoords[0] - mol.atcoords[2]) / angstrom, 0.9500, atol=1.0e-5
    )
    assert_allclose(
        np.linalg.norm(mol.atcoords[1] - mol.atcoords[2]) / angstrom, 1.5513, atol=1.0e-4
    )
    # check energies of scf cycles
    energies = np.array([-76.34739931, -76.34740001, -76.34740005, -76.34740029])
    assert_allclose(mol.extra["scf_energies"], energies)
    # check scf energy
    assert_allclose(mol.energy, -76.347791524303, atol=1e-8)
    # check dipole moment
    assert_allclose(mol.moments[(1, "c")], [0.76499, 0.00000, 0.54230])


def test_load_gradient():
    """Test loading ORCA output with atomic gradients."""
    with as_file(files("iodata.test.data").joinpath("orca_gradient.out")) as fn:
        mol = load_one(fn)
    # Test atomic numbers and number of atoms
    assert mol.natom == 32
    assert_equal(mol.atnums[:12], [6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1])
    # check scf energy
    assert_allclose(mol.energy, -742.985592886484, atol=1e-8)
    # check dipole moment
    assert_allclose(mol.moments[(1, "c")], [0.42692, 2.48902, 0.75997], atol=1e-5)
    # check atomic gradients
    assert mol.atgradient.shape == (32, 3)
    expected_gradients = np.array(
        [
            [-0.075712381, -0.005158513, 0.005897839],
            [0.029689323, -0.071778569, -0.025757841],
            [0.042197364, 0.025400096, 0.064826964],
            [0.005092660, 0.044375762, -0.047783083],
            [0.024156619, 0.052064005, 0.007957251],
            [0.004345097, 0.020061837, 0.002883165],
            [-0.030947207, -0.038254281, -0.011244397],
            [0.000214867, -0.011271100, -0.002943889],
            [0.032857036, -0.077160928, 0.007190076],
            [-0.025901011, 0.077214191, -0.000181281],
            [0.000509694, 0.014052826, -0.072295823],
            [-0.077223217, -0.005568882, 0.026475644],
        ]
    )
    assert_allclose(mol.atgradient[:12], expected_gradients[:12], atol=1e-8)

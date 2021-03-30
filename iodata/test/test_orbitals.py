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
# pylint: disable=pointless-statement
"""Unit tests for iodata.orbitals."""


import pytest
import numpy as np
from numpy.testing import assert_equal

from ..orbitals import MolecularOrbitals


def test_wrong_kind():
    with pytest.raises(ValueError):
        MolecularOrbitals("foo", 5, 5)


def test_restricted_empty():
    with pytest.raises(ValueError):
        MolecularOrbitals("restricted", 3, None)
    with pytest.raises(ValueError):
        MolecularOrbitals("restricted", None, 5)
    with pytest.raises(ValueError):
        MolecularOrbitals("restricted", None, None)
    with pytest.raises(ValueError):
        MolecularOrbitals("restricted", 5, 3)
    mo = MolecularOrbitals("restricted", 5, 5)
    assert mo.norba == 5
    assert mo.norbb == 5
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 5
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert mo.energiesa is None
    assert mo.energiesb is None
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_restricted_occs():
    occs = [2, 2, 0, 0, 0]
    with pytest.raises(TypeError):
        MolecularOrbitals("restricted", 3, 3, occs=occs)
    mo = MolecularOrbitals("restricted", 5, 5, occs=occs)
    assert mo.norba == 5
    assert mo.norbb == 5
    assert mo.nelec == 4
    assert mo.nbasis is None
    assert mo.norb == 5
    assert mo.spinpol == 0
    assert_equal(mo.occsa, [1, 1, 0, 0, 0])
    assert_equal(mo.occsb, [1, 1, 0, 0, 0])
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert mo.energiesa is None
    assert mo.energiesb is None
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_restricted_coeffs():
    coeffs = np.random.uniform(-1, 1, (7, 5))
    with pytest.raises(TypeError):
        MolecularOrbitals("restricted", 3, 3, coeffs=coeffs)
    mo = MolecularOrbitals("restricted", 5, 5, coeffs=coeffs)
    assert mo.norba == 5
    assert mo.norbb == 5
    assert mo.nelec is None
    assert mo.nbasis == 7
    assert mo.norb == 5
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert mo.coeffsa is coeffs
    assert mo.coeffsb is coeffs
    assert mo.energiesa is None
    assert mo.energiesb is None
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_restricted_energies():
    energies = np.random.uniform(-1, 1, 5)
    with pytest.raises(TypeError):
        MolecularOrbitals("restricted", 3, 3, energies=energies)
    mo = MolecularOrbitals("restricted", 5, 5, energies=energies)
    assert mo.norba == 5
    assert mo.norbb == 5
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 5
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert mo.energiesa is energies
    assert mo.energiesb is energies
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_restricted_irreps():
    irreps = ["A", "A", "B", "A", "B"]
    with pytest.raises(TypeError):
        MolecularOrbitals("restricted", 3, 3, irreps=irreps)
    mo = MolecularOrbitals("restricted", 5, 5, irreps=irreps)
    assert mo.norba == 5
    assert mo.norbb == 5
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 5
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert mo.energiesa is None
    assert mo.energiesb is None
    assert mo.irrepsa is irreps
    assert mo.irrepsb is irreps


def test_unrestricted_empty():
    with pytest.raises(ValueError):
        MolecularOrbitals("unrestricted", 3, None)
    with pytest.raises(ValueError):
        MolecularOrbitals("unrestricted", None, 5)
    with pytest.raises(ValueError):
        MolecularOrbitals("unrestricted", None, None)
    mo = MolecularOrbitals("unrestricted", 5, 3)
    assert mo.norba == 5
    assert mo.norbb == 3
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 8
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert mo.energiesa is None
    assert mo.energiesb is None
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_unrestricted_occs():
    occs = [1, 1, 0, 0, 0, 1, 0, 0]
    with pytest.raises(TypeError):
        MolecularOrbitals("unrestricted", 3, 2, occs=occs)
    mo = MolecularOrbitals("unrestricted", 5, 3, occs=occs)
    assert mo.norba == 5
    assert mo.norbb == 3
    assert mo.nelec == 3
    assert mo.nbasis is None
    assert mo.norb == 8
    assert mo.spinpol == 1
    assert_equal(mo.occsa, [1, 1, 0, 0, 0])
    assert_equal(mo.occsb, [1, 0, 0])
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert mo.energiesa is None
    assert mo.energiesb is None
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_unrestricted_coeffs():
    coeffs = np.random.uniform(-1, 1, (7, 8))
    with pytest.raises(TypeError):
        MolecularOrbitals("unrestricted", 3, 2, coeffs=coeffs)
    mo = MolecularOrbitals("unrestricted", 5, 3, coeffs=coeffs)
    assert mo.norba == 5
    assert mo.norbb == 3
    assert mo.nelec is None
    assert mo.nbasis == 7
    assert mo.norb == 8
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert_equal(mo.coeffsa, coeffs[:, :5])
    assert_equal(mo.coeffsb, coeffs[:, 5:])
    assert mo.energiesa is None
    assert mo.energiesb is None
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_unrestricted_energies():
    energies = np.random.uniform(-1, 1, 8)
    with pytest.raises(TypeError):
        MolecularOrbitals("unrestricted", 3, 2, energies=energies)
    mo = MolecularOrbitals("unrestricted", 5, 3, energies=energies)
    assert mo.norba == 5
    assert mo.norbb == 3
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 8
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert_equal(mo.energiesa, energies[:5])
    assert_equal(mo.energiesb, energies[5:])
    assert mo.irrepsa is None
    assert mo.irrepsb is None


def test_unrestricted_irreps():
    irreps = ["A", "A", "B", "A", "B", "B", "B", "A"]
    with pytest.raises(TypeError):
        MolecularOrbitals("unrestricted", 3, 2, irreps=irreps)
    mo = MolecularOrbitals("unrestricted", 5, 3, irreps=irreps)
    assert mo.norba == 5
    assert mo.norbb == 3
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 8
    assert mo.spinpol is None
    assert mo.occsa is None
    assert mo.occsb is None
    assert mo.coeffsa is None
    assert mo.coeffsb is None
    assert mo.energiesa is None
    assert mo.energiesb is None
    # irreps are lists, not arrays
    assert mo.irrepsa == irreps[:5]
    assert mo.irrepsb == irreps[5:]


def test_generalized_empty():
    with pytest.raises(ValueError):
        mo = MolecularOrbitals("generalized", 5, 3)
    with pytest.raises(ValueError):
        mo = MolecularOrbitals("generalized", 5, None)
    with pytest.raises(ValueError):
        mo = MolecularOrbitals("generalized", None, 3)
    mo = MolecularOrbitals("generalized", None, None)
    assert mo.norba is None
    assert mo.norbb is None
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb is None
    with pytest.raises(NotImplementedError):
        mo.spinpol
    with pytest.raises(NotImplementedError):
        mo.occsa
    with pytest.raises(NotImplementedError):
        mo.occsb
    with pytest.raises(NotImplementedError):
        mo.coeffsa
    with pytest.raises(NotImplementedError):
        mo.coeffsb
    with pytest.raises(NotImplementedError):
        mo.energiesa
    with pytest.raises(NotImplementedError):
        mo.energiesb
    with pytest.raises(NotImplementedError):
        mo.irrepsa
    with pytest.raises(NotImplementedError):
        mo.irrepsb


def test_generalized_occs():
    mo = MolecularOrbitals("generalized", None, None, occs=[1, 1, 1, 1, 1, 0, 0])
    assert mo.norba is None
    assert mo.norbb is None
    assert mo.nelec == 5
    assert mo.nbasis is None
    assert mo.norb == 7
    with pytest.raises(NotImplementedError):
        mo.spinpol
    with pytest.raises(NotImplementedError):
        mo.occsa
    with pytest.raises(NotImplementedError):
        mo.occsb
    with pytest.raises(NotImplementedError):
        mo.coeffsa
    with pytest.raises(NotImplementedError):
        mo.coeffsb
    with pytest.raises(NotImplementedError):
        mo.energiesa
    with pytest.raises(NotImplementedError):
        mo.energiesb
    with pytest.raises(NotImplementedError):
        mo.irrepsa
    with pytest.raises(NotImplementedError):
        mo.irrepsb


def test_generalized_coeffs():
    coeffs = np.random.uniform(-1, 1, (10, 7))
    mo = MolecularOrbitals("generalized", None, None, coeffs=coeffs)
    assert mo.norba is None
    assert mo.norbb is None
    assert mo.nelec is None
    assert mo.nbasis == 5  # 5 *spatial* basis functions!
    assert mo.norb == 7
    with pytest.raises(NotImplementedError):
        mo.spinpol
    with pytest.raises(NotImplementedError):
        mo.occsa
    with pytest.raises(NotImplementedError):
        mo.occsb
    with pytest.raises(NotImplementedError):
        mo.coeffsa
    with pytest.raises(NotImplementedError):
        mo.coeffsb
    with pytest.raises(NotImplementedError):
        mo.energiesa
    with pytest.raises(NotImplementedError):
        mo.energiesb
    with pytest.raises(NotImplementedError):
        mo.irrepsa
    with pytest.raises(NotImplementedError):
        mo.irrepsb


def test_generalized_energies():
    energies = np.random.uniform(-1, 1, 7)
    mo = MolecularOrbitals("generalized", None, None, energies=energies)
    assert mo.norba is None
    assert mo.norbb is None
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 7
    with pytest.raises(NotImplementedError):
        mo.spinpol
    with pytest.raises(NotImplementedError):
        mo.occsa
    with pytest.raises(NotImplementedError):
        mo.occsb
    with pytest.raises(NotImplementedError):
        mo.coeffsa
    with pytest.raises(NotImplementedError):
        mo.coeffsb
    with pytest.raises(NotImplementedError):
        mo.energiesa
    with pytest.raises(NotImplementedError):
        mo.energiesb
    with pytest.raises(NotImplementedError):
        mo.irrepsa
    with pytest.raises(NotImplementedError):
        mo.irrepsb


def test_generalized_irreps():
    irreps = ["A", "B", "A", "A", "B", "B", "B"]
    mo = MolecularOrbitals("generalized", None, None, irreps=irreps)
    assert mo.norba is None
    assert mo.norbb is None
    assert mo.nelec is None
    assert mo.nbasis is None
    assert mo.norb == 7
    with pytest.raises(NotImplementedError):
        mo.spinpol
    with pytest.raises(NotImplementedError):
        mo.occsa
    with pytest.raises(NotImplementedError):
        mo.occsb
    with pytest.raises(NotImplementedError):
        mo.coeffsa
    with pytest.raises(NotImplementedError):
        mo.coeffsb
    with pytest.raises(NotImplementedError):
        mo.energiesa
    with pytest.raises(NotImplementedError):
        mo.energiesb
    with pytest.raises(NotImplementedError):
        mo.irrepsa
    with pytest.raises(NotImplementedError):
        mo.irrepsb

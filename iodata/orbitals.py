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
"""Data structure for molecular orbitals."""


from typing import NamedTuple

import attr
import numpy as np


__all__ = ['RestrictedOrbitals', 'UnrestrictedOrbitals', 'GeneralizedOrbitals']


@attr.s(auto_attribs=True)
class Orbitals:
    """Molecular Orbitals base Class.

    Attributes
    ----------
    occs : np.ndarray
        Molecular orbital occupation numbers. The number of elements equals the
        number of columns of coeffs.
    coeffs : np.ndarray
        Molecular orbital basis coefficients.
        In case of restricted: shape = (nbasis, norb_a) = (nbasis, norb_b).
        In case of unrestricted: shape = (nbasis, norb_a + norb_b).
        In case of generalized: shape = (2*nbasis, norb), where norb is the
        total number of orbitals (not defined by other attributes).
    energies : np.ndarray
        Molecular orbital energies. The number of elements equals the
        number of columns of coeffs.
    irreps : np.ndarray
        Irreducible representation. The number of elements equals the
        number of columns of coeffs.

    """

    occs: np.ndarray
    coeffs: np.ndarray
    energies: np.ndarray
    irreps: np.ndarray

    @property
    def nelec(self) -> float:
        """Return the total number of electrons."""
        return self.occs.sum()

    @property
    def nbasis(self):
        """Return the number of spatial basis functions."""
        return self.coeffs.shape[0]

    @property
    def norb(self):
        """Return the number of orbitals."""
        return self.coeffs.shape[1]


@attr.s(auto_attribs=True)
class RestrictedOrbitals(Orbitals):
    """Restricted Orbitals class.

    Warning: the interpretation of the occupation numbers may only be suitable
    for single-reference orbitals (not fractionally occupied natural orbitals.)
    When an occupation number is in ]0, 1], it is assumed that an alpha orbital
    is (fractionally) occupied. When an occupation number is in ]1, 2], it is
    assumed that the alpha orbital is fully occupied and the beta orbital is
    (fractionally) occupied.
    """

    @property
    def spinpol(self) -> float:
        """Return the spin polarization of the Slater determinant."""
        nbeta = np.clip(self.occs, 0, 1).sum()
        return abs(self.nelec - 2 * nbeta)

    @property
    def occsa(self):
        """Return alpha occupation numbers."""
        return np.clip(self.occs, 0, 1)

    @property
    def occsb(self):
        """Return beta occupation numbers."""
        return self.occs - np.clip(self.occs, 0, 1)

    @property
    def coeffsa(self):
        """Return alpha orbital coefficients."""
        return self.coeffs

    @property
    def coeffsb(self):
        """Return beta orbital coefficients."""
        return self.coeffs

    @property
    def irrepsa(self):
        """Return alpha irreps."""
        return self.irreps

    @property
    def irrepsb(self):
        """Return beta irreps."""
        return self.irreps

    @property
    def energiesa(self):
        """Return alpha orbital energies."""
        return self.energies

    @property
    def energiesb(self):
        """Return beta orbital energies."""
        return self.energies


@attr.s(auto_attribs=True)
class UnrestrictedOrbitals(Orbitals):
    """Unestricted Orbitals class.

    New attributes not defined in Orbitals
    --------------------------------------
    norba : int
        Number of alpha molecular orbitals. Only present in case of

    """

    # Note: one cannot inherit attributes of a NamedTuple. They must be defined
    # in the subclass.
    norba: int
    occs: np.ndarray
    coeffs: np.ndarray
    energies: np.ndarray
    irreps: np.ndarray

    @property
    def norbb(self):
        """Return the number of beta orbitals."""
        return self.norb - self.norba

    @norbb.setter
    def norbb(self, norbb):
        self.norba = self.norb - norbb

    @property
    def spinpol(self) -> float:
        """Return the spin polarization of the Slater determinant."""
        return abs(self.occsa.sum() - self.occsb.sum())

    @property
    def occsa(self):
        """Return alpha occupation numbers."""
        return self.occs[:self.norba]

    @property
    def occsb(self):
        """Return beta occupation numbers."""
        return self.occs[self.norba:]

    @property
    def coeffsa(self):
        """Return alpha orbital coefficients."""
        return self.coeffs[:, :self.norba]

    @property
    def coeffsb(self):
        """Return beta orbital coefficients."""
        return self.coeffs[:, self.norba:]

    @property
    def irrepsa(self):
        """Return alpha irreps."""
        return self.irreps[:self.norba]

    @property
    def irrepsb(self):
        """Return beta irreps."""
        return self.irreps[self.norba:]

    @property
    def energiesa(self):
        """Return alpha orbital energies."""
        return self.energies[:self.norba]

    @property
    def energiesb(self):
        """Return beta orbital energies."""
        return self.energies[self.norba:]


@attr.s(auto_attribs=True)
class GeneralizedOrbitals(Orbitals, NamedTuple):
    """Generalized Orbitals class."""

    @property
    def nbasis(self):
        """Return the number of spatial basis functions."""
        return self.coeffs.shape[0] // 2

    @property
    def spinpol(self) -> float:
        """Return the spin polarization of the Slater determinant."""
        raise NotImplementedError

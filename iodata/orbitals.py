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

import numpy as np


__all__ = ['MolecularOrbitals']


class MolecularOrbitals(NamedTuple):
    """Class of Orthonormal Molecular Orbitals.

    Attributes
    ----------
    kind
        Type of molecular orbitals, which can be 'restricted', 'unrestricted', or 'generalized'.
    norba
        Number of alpha molecular orbitals, set to `None` in case of type=='generalized'.
    norbb
        Number of beta molecular orbitals, set to `None` in case of type=='generalized'.
        This is expected to be equal to `norba` for the `restricted` kind.
    occs
        Molecular orbital occupation numbers. The length equals the number of columns of coeffs.
    coeffs
        Molecular orbital coefficients.
        In case of restricted: shape = (nbasis, norba) = (nbasis, norbb).
        In case of unrestricted: shape = (nbasis, norba + norbb).
        In case of generalized: shape = (2 * nbasis, norb), where norb is the
        total number of orbitals.
    energies
        Molecular orbital energies. The length equals the number of columns of coeffs.
    irreps
        Irreducible representation. The length equals the number of columns of coeffs.

    Warning: the interpretation of the occupation numbers may only be suitable
    for single-reference orbitals (not fractionally occupied natural orbitals.)
    When an occupation number is in ]0, 1], it is assumed that an alpha orbital
    is (fractionally) occupied. When an occupation number is in ]1, 2], it is
    assumed that the alpha orbital is fully occupied and the beta orbital is
    (fractionally) occupied.

    """

    kind: str
    norba: int
    norbb: int
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
        if self.kind == 'generalized':
            return self.coeffs.shape[0] // 2
        return self.coeffs.shape[0]

    @property
    def norb(self):
        """Return the number of orbitals."""
        return self.coeffs.shape[1]

    @property
    def spinpol(self) -> float:
        """Return the spin polarization of the Slater determinant."""
        if self.kind == 'restricted':
            nbeta = np.clip(self.occs, 0, 1).sum()
            return abs(self.nelec - 2 * nbeta)
        if self.kind == 'unrestricted':
            return abs(self.occsa.sum() - self.occsb.sum())
        raise NotImplementedError

    @property
    def occsa(self):
        """Return alpha occupation numbers."""
        if self.kind == 'restricted':
            return np.clip(self.occs, 0, 1)
        if self.kind == 'unrestricted':
            return self.occs[:self.norba]
        raise NotImplementedError

    @property
    def occsb(self):
        """Return beta occupation numbers."""
        if self.kind == 'restricted':
            return self.occs - np.clip(self.occs, 0, 1)
        if self.kind == 'unrestricted':
            return self.occs[self.norba:]
        raise NotImplementedError

    @property
    def coeffsa(self):
        """Return alpha orbital coefficients."""
        if self.kind == 'restricted':
            return self.coeffs
        if self.kind == 'unrestricted':
            return self.coeffs[:, :self.norba]
        raise NotImplementedError

    @property
    def coeffsb(self):
        """Return beta orbital coefficients."""
        if self.kind == 'restricted':
            return self.coeffs
        if self.kind == 'unrestricted':
            return self.coeffs[:, self.norba:]
        raise NotImplementedError

    @property
    def irrepsa(self):
        """Return alpha irreps."""
        if self.irreps is None:
            return None
        if self.kind == 'restricted':
            return self.irreps
        if self.kind == 'unrestricted':
            return self.irreps[:self.norba]
        raise NotImplementedError

    @property
    def irrepsb(self):
        """Return beta irreps."""
        if self.irreps is None:
            return None
        if self.kind == 'restricted':
            return self.irreps
        if self.kind == 'unrestricted':
            return self.irreps[self.norba:]
        raise NotImplementedError

    @property
    def energiesa(self):
        """Return alpha orbital energies."""
        if self.kind == 'restricted':
            return self.energies
        if self.kind == 'unrestricted':
            return self.energies[:self.norba]
        raise NotImplementedError

    @property
    def energiesb(self):
        """Return beta orbital energies."""
        if self.kind == 'restricted':
            return self.energies
        if self.kind == 'unrestricted':
            return self.energies[self.norba:]
        raise NotImplementedError


MolecularOrbitals.__defaults__ = (None,)

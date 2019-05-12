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
    """Molecular Orbitals Class.

    Attributes
    ----------
    type : str
        Molecular orbital type; choose from 'restricted', 'unrestricted', or 'generalized'.
    norba : int
        Number of alpha molecular orbitals. None in case of type=='generalized'.
    norbb : int
        Number of beta molecular orbitals. None in case of type=='generalized'.
    occs : np.ndarray
        Molecular orbital occupation numbers. The number of elements equals the
        number of columns of coeffs.
    coeffs : np.ndarray
        Molecular orbital basis coefficients.
        In case of restricted: shape = (nbasis, norb_a) = (nbasis, norb_b).
        In case of unrestricted: shape = (nbasis, norb_a + norb_b).
        In case of generalized: shape = (2*nbasis, norb), where norb is the
        total number of orbitals (not defined by other attributes).
    irreps : np.ndarray
        Irreducible representation. The number of elements equals the
        number of columns of coeffs.
    energies : np.ndarray
        Molecular orbital energies. The number of elements equals the
        number of columns of coeffs.

    """

    type: str
    norba: int
    norbb: int
    occs: np.ndarray
    coeffs: np.ndarray
    irreps: np.ndarray
    energies: np.ndarray

    @property
    def nelec(self) -> float:
        """Return the total number of electrons."""
        return self.occs.sum()

    @property
    def spinpol(self) -> float:
        """Return the spin multiplicity of the Slater determinant."""
        if self.type == 'restricted':
            nbeta = np.clip(self.occs, 0, 1).sum()
            sq = self.nelec - 2 * nbeta
        elif self.type == 'unrestricted':
            sq = self.occs[:self.norba].sum() - self.occs[self.norba:].sum()
        else:
            # Not sure how to do this in a simply way.
            raise NotImplementedError
        return abs(sq)

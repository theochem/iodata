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


import attr
import numpy as np

from .attrutils import convert_array_to, validate_shape


__all__ = ['MolecularOrbitals']


def validate_norbab(mo, attribute, value):
    """Validate the norba or norbb value assigned to a MolecularOrbitals object.

    Parameters
    ----------
    mo
        The MolecularOrbitals instance.
    attribute
        Attribute instancce being changed.
    value
        The new value.

    """
    if mo.kind == "generalized":
        if value is not None:
            raise ValueError(
                f"Attribute {attribute.name} must be None in case of generalized orbitals.")
        return
    if value is None:
        raise ValueError(
            f"Attribute {attribute.name} cannot be None in case of (un)restricted orbitals.")
    if mo.kind == "restricted":
        norb_other = mo.norbb if (attribute.name == "norba") else mo.norba
        if value != norb_other:
            raise ValueError("In case of restricted orbitals, norba must be equal to norbb.")


@attr.s(auto_attribs=True, slots=True,
        on_setattr=[attr.setters.validate, attr.setters.convert])
class MolecularOrbitals:
    """Class of Orthonormal Molecular Orbitals.

    Attributes
    ----------
    kind
        Type of molecular orbitals, which can be 'restricted', 'unrestricted', or 'generalized'.
    norba
        Number of (occupied and virtual) alpha molecular orbitals.
        Set to `None` in case oftype=='generalized'.
    norbb
        Number of (occupied and virtual) beta molecular orbitals.
        Set to `None` in case of type=='generalized'.
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

    kind: str = attr.ib(
        validator=attr.validators.in_(["restricted", "unrestricted", "generalized"]))
    norba: int = attr.ib(validator=validate_norbab)
    norbb: int = attr.ib(validator=validate_norbab)
    occs: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=attr.validators.optional(validate_shape("norb")))
    coeffs: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=attr.validators.optional(validate_shape(None, "norb")))
    energies: np.ndarray = attr.ib(
        default=None, converter=convert_array_to(float),
        validator=attr.validators.optional(validate_shape("norb")))
    irreps: np.ndarray = attr.ib(
        default=None,
        validator=attr.validators.optional(validate_shape("norb")))

    @property
    def nelec(self) -> float:
        """Return the total number of electrons."""
        if self.occs is None:
            return None
        return self.occs.sum()

    @property
    def nbasis(self):
        """Return the number of spatial basis functions."""
        if self.coeffs is None:
            return None
        if self.kind == 'generalized':
            return self.coeffs.shape[0] // 2
        return self.coeffs.shape[0]

    @property
    def norb(self):  # pylint: disable=too-many-return-statements
        """Return the number of spatially distinct orbitals.

        Notes
        -----
        In case of restricted wavefunctions, this may be less than just the
        sum of ``norba`` and ``norbb``, because alpha and beta orbitals share
        the same spatical dependence.

        """
        if self.kind == "restricted":
            return self.norba
        if self.kind == "unrestricted":
            return self.norba + self.norbb
        if self.coeffs is not None:
            return self.coeffs.shape[1]
        if self.occs is not None:
            return self.occs.shape[0]
        if self.energies is not None:
            return self.energies.shape[0]
        if self.irreps is not None:
            return len(self.irreps)
        return None

    @property
    def spinpol(self) -> float:
        """Return the spin polarization of the Slater determinant."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.occs is None:
            return None
        if self.kind == 'restricted':
            nbeta = np.clip(self.occs, 0, 1).sum()
            return abs(self.nelec - 2 * nbeta)
        return abs(self.occsa.sum() - self.occsb.sum())

    @property
    def occsa(self):
        """Return alpha occupation numbers."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.occs is None:
            return None
        if self.kind == 'restricted':
            return np.clip(self.occs, 0, 1)
        return self.occs[:self.norba]

    @property
    def occsb(self):
        """Return beta occupation numbers."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.occs is None:
            return None
        if self.kind == 'restricted':
            return self.occs - np.clip(self.occs, 0, 1)
        return self.occs[self.norba:]

    @property
    def coeffsa(self):
        """Return alpha orbital coefficients."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.coeffs is None:
            return None
        if self.kind == 'restricted':
            return self.coeffs
        return self.coeffs[:, :self.norba]

    @property
    def coeffsb(self):
        """Return beta orbital coefficients."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.coeffs is None:
            return None
        if self.kind == 'restricted':
            return self.coeffs
        return self.coeffs[:, self.norba:]

    @property
    def energiesa(self):
        """Return alpha orbital energies."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.energies is None:
            return None
        if self.kind == 'restricted':
            return self.energies
        return self.energies[:self.norba]

    @property
    def energiesb(self):
        """Return beta orbital energies."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.energies is None:
            return None
        if self.kind == 'restricted':
            return self.energies
        return self.energies[self.norba:]

    @property
    def irrepsa(self):
        """Return alpha irreps."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.irreps is None:
            return None
        if self.kind == 'restricted':
            return self.irreps
        return self.irreps[:self.norba]

    @property
    def irrepsb(self):
        """Return beta irreps."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.irreps is None:
            return None
        if self.kind == 'restricted':
            return self.irreps
        return self.irreps[self.norba:]

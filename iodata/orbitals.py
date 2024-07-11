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

from typing import Optional

import attrs
import numpy as np
from numpy.typing import NDArray

from .attrutils import convert_array_to, validate_shape

__all__ = ("MolecularOrbitals",)


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
                f"Attribute {attribute.name} must be None in case of generalized orbitals."
            )
        return
    if value is None:
        raise ValueError(
            f"Attribute {attribute.name} cannot be None in case of (un)restricted orbitals."
        )
    if mo.kind == "restricted":
        norb_other = mo.norbb if (attribute.name == "norba") else mo.norba
        if value != norb_other:
            raise ValueError("In case of restricted orbitals, norba must be equal to norbb.")


def validate_occs_aminusb(mo, _attribtue, value):
    """Validate the occs_aminusb attribute."""
    if mo.kind != "restricted" and value is not None:
        raise ValueError("Attribute occs_aminusb can only be set for restricted wavefunctions.")


@attrs.define
class MolecularOrbitals:
    """Class of Orthonormal Molecular Orbitals.

    Notes
    -----

    For restricted wavefunctions, the occupation numbers are spin-summed values
    and several rules are used to deduce the alpha and beta occupation
    numbers:

    - When ``occs_aminusb`` is set, alpha and beta occupation numbers are
      derived trivially as ``(occs + occs_aminusb) / 2`` and
      ``(occs - occs_aminusb) / 2``, respectively.

    - When ``occs_aminusb`` is not set, there are two possibilities. When the
      occupation numbers are integers, it is assumed that the orbitals represent
      a restricted open-shell HF or KS wavefunction. An occupation number of 1
      is then interpreted as an occupied alpha orbital and a virtual beta
      orbital. When the occupation numbers are fractional, it is assumed that
      the orbitals are closed-shell natural orbitals.

    One can always describe all cases by setting ``occs_aminusb``. While this
    seems appealing, keep in mind that most wavefunction file formats (FCHK,
    Molden, Molekel, WFN and WFX) do not support it.

    """

    kind: str = attrs.field(
        validator=attrs.validators.in_(["restricted", "unrestricted", "generalized"])
    )
    """Type of molecular orbitals, which can be 'restricted', 'unrestricted', or 'generalized'."""

    norba: int = attrs.field(validator=validate_norbab)
    """
    Number of (occupied and virtual) alpha molecular orbitals.
    Set to `None` in case oftype=='generalized'.
    """

    norbb: int = attrs.field(validator=validate_norbab)
    """
    Number of (occupied and virtual) beta molecular orbitals.
    Set to `None` in case of type=='generalized'.
    This is expected to be equal to `norba` for the `restricted` kind.
    """

    occs: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape("norb")),
    )
    """
    Molecular orbital occupation numbers.
    The length equals the number of columns of coeffs. (optional)
    """

    coeffs: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape(None, "norb")),
    )
    """
    Molecular orbital coefficients.
    In case of restricted: shape = (nbasis, norba) = (nbasis, norbb).
    In case of unrestricted: shape = (nbasis, norba + norbb).
    In case of generalized: shape = (2 * nbasis, norb), where norb is the
    total number of orbitals. (optional)
    """

    energies: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape("norb")),
    )
    """Molecular orbital energies. The length equals the number of columns of coeffs. (optional)"""

    irreps: Optional[NDArray] = attrs.field(
        default=None, validator=attrs.validators.optional(validate_shape("norb"))
    )
    """Irreducible representation. The length equals the number of columns of coeffs. (optional)"""

    occs_aminusb: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.and_(
            attrs.validators.optional(validate_shape("norb")), validate_occs_aminusb
        ),
    )
    """
    The difference between alpha and beta occupation numbers.
    The length equals the number of columns of coeffs.
    (optional and only allowed to be not None for restricted wavefunctions)
    """

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
        if self.kind == "generalized":
            return self.coeffs.shape[0] // 2
        return self.coeffs.shape[0]

    @property
    def norb(self):
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
    def spinpol(self) -> Optional[float]:
        """Return the spin polarization of the Slater determinant."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.occs is None:
            return None
        if self.kind == "restricted":
            if self.occs_aminusb is None:
                # heuristics ...
                if (self.occs == self.occs.astype(int)).all():
                    # restricted open-shell HF/KS
                    nbeta = np.clip(self.occs, 0, 1).sum()
                    return abs(self.nelec - 2 * nbeta)
                # restricted closed-shell natural orbitals
                return 0.0
            return self.occs_aminusb.sum()
        return abs(self.occsa.sum() - self.occsb.sum())

    @property
    def occsa(self):
        """Return alpha occupation numbers.

        Notes
        -----
        For restricted wavefunctions, in-place assignment to occsa will not
        work. In this case, the array is derived from ``mo.occs`` and optionally
        ``mo.occs_aminusb``. To avoid that in-place assignment of occsa is
        silently ignored, it is returned as a non-writeable array. To change
        occsa, one can assign a whole new array, e.g. ``mo.occsa = new_occsa``
        will work, while ``mo.occsa[1] = 0.3`` will not.

        """
        if self.kind == "generalized":
            raise NotImplementedError
        if self.occs is None:
            return None
        if self.kind == "restricted":
            if self.occs_aminusb is None:
                # heuristics ...
                if (self.occs == self.occs.astype(int)).all():
                    # restricted open-shell HF/KS
                    result = np.clip(self.occs, 0, 1)
                else:
                    # restricted closed-shell natural orbitals
                    result = self.occs / 2
            else:
                result = (self.occs + self.occs_aminusb) / 2
            result.flags.writeable = False
            return result
        return self.occs[: self.norba]

    @occsa.setter
    def occsa(self, occsa):
        if self.kind == "generalized":
            raise NotImplementedError
        if self.kind == "restricted":
            occsa = np.array(occsa)
            if self.occs is None:
                self.occs = occsa
                self.occs_aminusb = occsa.copy()
            else:
                occsb = np.array(self.occsb)
                self.occs = occsa + occsb
                self.occs_aminusb = occsa - occsb
        else:
            self.occs[: self.norba] = occsa

    @property
    def occsb(self):
        """Return beta occupation numbers.

        Notes
        -----
        For restricted wavefunctions, in-place assignment to occsb will not
        work. In this case, the array is derived from ``mo.occs`` and optionally
        ``mo.occs_aminusb``. To avoid that in-place assignment of occsb is
        silently ignored, it is returned as a non-writeable array. To change
        occsb, one can assign a whole new array, e.g. ``mo.occsb = new_occsb``
        will work, while ``mo.occsb[1] = 0.3`` will not.

        """
        if self.kind == "generalized":
            raise NotImplementedError
        if self.occs is None:
            return None
        if self.kind == "restricted":
            if self.occs_aminusb is None:
                # heuristics ...
                if (self.occs == self.occs.astype(int)).all():
                    # restricted open-shell HF/KS
                    result = self.occs - np.clip(self.occs, 0, 1)
                else:
                    # restricted closed-shell natural orbitals
                    result = self.occs / 2
            else:
                result = (self.occs - self.occs_aminusb) / 2
            result.flags.writeable = False
            return result
        return self.occs[self.norba :]

    @occsb.setter
    def occsb(self, occsb):
        if self.kind == "generalized":
            raise NotImplementedError
        if self.kind == "restricted":
            occsb = np.array(occsb)
            if self.occs is None:
                self.occs = occsb
                self.occs_aminusb = -occsb
            else:
                occsa = np.array(self.occsa)
                self.occs = occsa + occsb
                self.occs_aminusb = occsa - occsb
        else:
            self.occs[self.norba :] = occsb

    @property
    def coeffsa(self):
        """Return alpha orbital coefficients."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.coeffs is None:
            return None
        if self.kind == "restricted":
            return self.coeffs
        return self.coeffs[:, : self.norba]

    @property
    def coeffsb(self):
        """Return beta orbital coefficients."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.coeffs is None:
            return None
        if self.kind == "restricted":
            return self.coeffs
        return self.coeffs[:, self.norba :]

    @property
    def energiesa(self):
        """Return alpha orbital energies."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.energies is None:
            return None
        if self.kind == "restricted":
            return self.energies
        return self.energies[: self.norba]

    @property
    def energiesb(self):
        """Return beta orbital energies."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.energies is None:
            return None
        if self.kind == "restricted":
            return self.energies
        return self.energies[self.norba :]

    @property
    def irrepsa(self):
        """Return alpha irreps."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.irreps is None:
            return None
        if self.kind == "restricted":
            return self.irreps
        return self.irreps[: self.norba]

    @property
    def irrepsb(self):
        """Return beta irreps."""
        if self.kind == "generalized":
            raise NotImplementedError
        if self.irreps is None:
            return None
        if self.kind == "restricted":
            return self.irreps
        return self.irreps[self.norba :]

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
"""Module for handling input/output from different file formats."""

from typing import Optional

import attrs
import numpy as np
from numpy.typing import NDArray

from .attrutils import convert_array_to, validate_shape
from .basis import MolecularBasis
from .orbitals import MolecularOrbitals
from .utils import Cube

__all__ = ("IOData",)


@attrs.define
class IOData:
    """A container class for data loaded from (or to be written to) a file."""

    atcharges: dict = attrs.field(factory=dict)
    """
    A dictionary where keys are names of charge definitions and
    values are arrays with atomic charges (size N).
    """

    atcoords: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape("natom", 3)),
    )
    """A (N, 3) float array with Cartesian coordinates of the atoms."""

    _atcorenums: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape("natom")),
    )
    """
    A (N,) float array with pseudo-potential core charges.
    The matrix elements corresponding to ghost atoms are zero.
    """

    atffparams: dict = attrs.field(factory=dict)
    """
    A dictionary with arrays of atomic force field parameters (typically
    non-bonded). Keys include 'charges', 'vdw_radii', 'sigmas', 'epsilons',
    'alphas' (atomic polarizabilities), 'c6s', 'c8s', 'c10s', 'buck_as',
    'buck_bs', 'lj_as', 'core_charges', 'valence_charges', 'valence_widths',
    etc. Not all of them have to be present, depending on the use case.
    """

    atfrozen: Optional[NDArray[bool]] = attrs.field(
        default=None,
        converter=convert_array_to(bool),
        validator=attrs.validators.optional(validate_shape("natom")),
    )
    """A (N,) bool array with frozen atoms. (All atoms are free if thisattribute is not set.)"""

    atgradient: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape("natom", 3)),
    )
    """
    A (N, 3) float array with the first derivatives of the energy w.r.t.
    Cartesian atomic displacements.
    """

    athessian: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape(None, None)),
    )
    """A (3*N, 3*N) array containing the energy Hessian w.r.t Cartesian atomic displacements."""

    atmasses: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape("natom")),
    )
    """A (N,) float array with atomic masses."""

    atnums: Optional[NDArray[int]] = attrs.field(
        default=None,
        converter=convert_array_to(int),
        validator=attrs.validators.optional(validate_shape("natom")),
    )
    """A (N,) int vector with the atomic numbers."""

    basisdef: Optional[str] = attrs.field(default=None)
    """
    A basis set definition, i.e. a dictionary whose keys are symbols (of
    chemical elements), atomic numbers (similar to previous, str to make
    distinction with following) or an atom index (integer referring to a
    specific atom in a molecule). The format of the values is to be decided
    when implementing a load function for basis set definitions.
    """

    bonds: Optional[NDArray[int]] = attrs.field(
        default=None,
        converter=convert_array_to(int),
        validator=attrs.validators.optional(validate_shape(None, 3)),
    )
    """
    An (nbond, 3) array with the list of covalent bonds. Each row represents
    one bond and consists of three integers: first atom index (starting
    from zero), second atom index & an optional bond type. Numerical values
    of bond types are defined in ``iodata.periodic``.
    """

    cellvecs: Optional[NDArray[float]] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape(None, 3)),
    )
    """
    A (NP, 3) array with (real-space) cell vectors describing periodic boundary conditions.
    A single vector corresponds to a 1D cell, e.g. for a wire.
    Two vectors describe a 2D cell, e.g. for a membrane.
    Three vectors describe a 3D cell, e.g. a crystalline solid.
    """

    _charge: Optional[float] = attrs.field(default=None)
    """The net charge of the system. When possible, this is derived from atcorenums and nelec."""

    core_energy: Optional[float] = attrs.field(default=None)
    """The Hartree-Fock energy due to the core orbitals."""

    cube: Optional[Cube] = attrs.field(default=None)
    """An instance of Cube, describing the volumetric data from a cube (or similar) file."""

    energy: Optional[float] = attrs.field(default=None)
    """The total energy (electronic + nn)."""

    extcharges: NDArray[float] = attrs.field(
        default=None,
        converter=convert_array_to(float),
        validator=attrs.validators.optional(validate_shape(None, 4)),
    )
    """
    Array with values of external charges, with shape (nextcharge, 4).
    First three columns for Cartesian X, Y and Z coordinates, last column for the actual charge.
    """

    extra: dict = attrs.field(factory=dict)
    """
    A dictionary with additional data loaded from a file.
    Any data which cannot be assigned to the other attributes belongs here.
    It may be decided in future to move some of the results from this dictionary
    to IOData attributes, with a more final name.
    """

    g_rot: Optional[float] = attrs.field(default=None)
    """The rotational symmetry number of the molecule."""

    lot: Optional[str] = attrs.field(default=None)
    """The level of theory used to compute the orbitals (and other properties)."""

    mo: Optional[MolecularOrbitals] = attrs.field(default=None)
    """The molecular orbitals."""

    moments: dict = attrs.field(factory=dict)
    """
    A dictionary with electrostatic multipole moments.
    Keys are (angmom, kind) tuples where angmom is an integer for the angular momentum and
    kind is 'c' for Cartesian or 'p' for pure functions (only for angmom >= 2).
    The corresponding values are 1D numpy arrays.
    The order of the components of the multipole moments follows the HORTON2_CONVENTIONS
    defined in :py:mod:`iodata.basis`.
    """

    _nelec: Optional[float] = attrs.field(default=None)
    """The number of electrons."""

    obasis: Optional[MolecularBasis] = attrs.field(default=None)
    """An OrderedDict containing parameters to instantiate a GOBasis class."""

    obasis_name: Optional[str] = attrs.field(default=None)
    """
    A name or DOI describing the basis set used for the orbitals in the mo attribute (if defined).
    The name should be consistent with those defined the
    `Basis Set Exchange <www.basissetexchange.org>`_.
    """

    one_ints: dict = attrs.field(factory=dict)
    """
    Dictionary where keys are names and values are numpy arrays with one-body operators,
    typically integrals of a one-body operator with a pair of (Gaussian) basis functions.
    Names can start with ``olp`` (overlap), ``kin`` (kinetic energy), ``na`` (nuclear attraction),
    ``core`` (core hamiltonian), etc., or ``one`` (general one-electron integral).
    When relevant, these names must have a suffix ``_ao`` or ``_mo`` to clarify in which basis
    the integrals are computed.
    ``_ao`` is used to denote integrals in a non-orthogonal (atomic orbital) basis.
    ``_mo`` is used to denote an orthogonal (molecular orbital) basis.
    For the overlap integrals, this suffix can be omitted
    because it is only useful to compute them in the atomic-orbital basis.
    """

    one_rdms: dict = attrs.field(factory=dict)
    """
    Dictionary where keys are names and values are one-particle density matrices.
    Names can be ``scf``, ``post_scf``, ``scf_spin``, ``post_scf_spin``.
    When relevant, these names must have a suffix ``_ao`` or ``_mo``
    to clarify in which basis the RDMs are computed.
    ``_ao`` is used to denote a non-orthogonal (atomic orbital) basis.
    ``_mo`` is used to denote an orthogonal (molecular orbital) basis.
    For the SCF RDMs, this suffix can be omitted
    because it is only useful to compute them in the atomic-orbital basis.
    """

    run_type: Optional[str] = attrs.field(default=None)
    """
    The type of calculation that lead to the results stored in IOData,
    which must be one of the following: 'energy', 'energy_force', 'opt', 'scan', 'freq' or None.
    """

    _spinpol: Optional[float] = attrs.field(default=None)
    """
    The spin polarization.
    When molecular orbitals are present,
    spinpol cannot be set and is derived as ``abs(nalpha - nbeta)``.
    When no molecular orbitals are present, this attribute can be set.
    """

    title: Optional[str] = attrs.field(default=None)
    """A suitable name for the data."""

    two_ints: dict = attrs.field(factory=dict)
    """
    Dictionary where keys are names and values are numpy arrays with two-body operators,
    typically integrals of two-body operator with four of (Gaussian) basis functions.
    Names can start with ``er`` (electron repulsion) or ``two`` (general pairswise interaction).
    When relevant, these names must have a suffix ``_ao`` or ``_mo`` to clarify in which
    basis the integrals are computed.
    See ``one_ints`` for more details.
    Array indexes are in physicists' notation.
    """

    two_rdms: dict = attrs.field(factory=dict)
    """
    Dictionary where keys are names and values are two-particle density matrices.
    Names can be ``post_scf`` or ``post_scf_spin``.
    When relevant, these names must have a suffix ``_ao`` or ``_mo`` to clarify in which
    basis the RDMs are computed.
    See ``one_rdms`` for more details.
    Array indexes are in physicists' notation.
    """

    def __attrs_post_init__(self):
        # Trigger setter to acchieve consistency in properties
        # atcorenums, nelec, charge, spinmult. This is needed because the
        # attr constructor bypasses the setters. See
        # https://www.attrs.org/en/stable/init.html#private-attributes
        if self._atcorenums is not None:
            self.atcorenums = self._atcorenums
        if self._charge is not None:
            self.charge = self._charge
        if self._nelec is not None:
            self.nelec = self._nelec
        if self._spinpol is not None:
            self.spinpol = self._spinpol

    # Public interfaces to private attributes

    @property
    def atcorenums(self) -> NDArray[float]:
        """Effective core charges."""
        if self._atcorenums is None and self.atnums is not None:
            self.atcorenums = self.atnums.astype(float)
        return self._atcorenums

    @atcorenums.setter
    def atcorenums(self, atcorenums: NDArray):
        if atcorenums is None:
            if self.nelec is not None and self._atcorenums is not None:
                # Set _charge because charge can no longer be derived from
                # atcorenums and nelec.
                self._charge = self._atcorenums.sum() - self.nelec
            self._atcorenums = None
        else:
            if self._charge is not None:
                # _charge is treated as the dependent one, while atcorenums and
                # nelec are treated as independent.
                if self._nelec is None:
                    # Switch to storing _nelec.
                    self._nelec = atcorenums.sum() - self._charge
                self._charge = None
            self._atcorenums = np.asarray(atcorenums, dtype=float)

    @property
    def charge(self) -> float:
        """Return the net charge of the system."""
        # The internal _charge is used only if it cannot be derived.
        if self.atcorenums is None or self.nelec is None:
            return self._charge
        return self.atcorenums.sum() - self.nelec

    @charge.setter
    def charge(self, charge: float):
        # The internal _charge is used only if atcorenums is None.
        if self.atcorenums is None:
            self._charge = charge
        elif charge is None:
            self.nelec = None
        else:
            self.nelec = self.atcorenums.sum() - charge

    @property
    def natom(self) -> int:
        """Return the number of atoms."""
        natom = None
        if self.atcoords is not None:
            natom = len(self.atcoords)
        elif self._atcorenums is not None:
            natom = len(self._atcorenums)
        elif self.atgradient is not None:
            natom = len(self.atgradient)
        elif self.atfrozen is not None:
            natom = len(self.atfrozen)
        elif self.atmasses is not None:
            natom = len(self.atmasses)
        elif self.atnums is not None:
            natom = len(self.atnums)
        return natom

    @property
    def nelec(self) -> float:
        """Return the number of electrons."""
        # When the molecular orbitals are present, they determine the number
        # of electrons. Only when mo is absent, we use a stored value.
        if self.mo is not None:
            return self.mo.nelec
        return self._nelec

    @nelec.setter
    def nelec(self, nelec: float):
        if self.mo is None:
            self._nelec = nelec
        else:
            raise TypeError("nelec cannot be set when orbitals are present.")

    @property
    def spinpol(self) -> float:
        """Return the spin polarization.

        Warning: for restricted wavefunctions, it is assumed that an occupation
        number in ]0, 2[ implies spin polarizaiton, which may not always be a
        valid assumption.
        """
        if self.mo is not None:
            return self.mo.spinpol
        return self._spinpol

    @spinpol.setter
    def spinpol(self, spinpol: float):
        # When the molecular orbitals are present, they determine the spin
        # polarization. Only when mo is absent, we use a stored value.
        if self.mo is None:
            self._spinpol = spinpol
        else:
            raise TypeError("spinpol cannot be set when orbitals are present.")
